# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading worker — direct GPU<->NVMe via cuFile sync API."""

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.gds.common import GDSLoadStoreSpec
from vllm.v1.kv_offload.gds.cufile_bindings import (
    CUfileHandle_t,
    cuFileDriverClose,
    cuFileDriverOpen,
    cuFileHandleDeregister,
    cuFileHandleRegister,
    cuFileRead,
    cuFileWrite,
    open_for_gds,
)

logger = init_logger(__name__)

DEFAULT_MAX_THREADS = 400


@dataclass
class _Transfer:
    job_id: int
    futures: list[Future]
    num_bytes: int
    start_time: float
    file_handles: list[tuple[CUfileHandle_t, int]] = field(default_factory=list)


class GDSOffloadingWorker(OffloadingWorker):
    """GPU<->NVMe transfers via NVIDIA cuFile synchronous API.

    Each file's I/O is submitted to a thread pool so multiple operations
    overlap on the NVMe device (higher queue depth = higher throughput).
    Completion is tracked via futures.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        file_mapper: FileMapper,
        max_io_threads: int = DEFAULT_MAX_THREADS,
    ):
        self._file_mapper = file_mapper
        self._block_size_factor = block_size_factor
        self._kv_cache_groups_data_refs = kv_caches.group_data_refs

        # Initialize cuFile driver
        cuFileDriverOpen()
        logger.info("cuFile driver opened")

        # Build views of KV cache tensors for block-level addressing
        self._gpu_tensors: list[torch.Tensor] = []

        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            self._gpu_tensors.append(gpu_tensor)

        # Thread pool for sync cuFile I/O
        self._pool = ThreadPoolExecutor(max_workers=max_io_threads)
        logger.info("GDS worker thread pool: max_workers=%d", max_io_threads)

        # In-flight transfers
        self._transfers: dict[int, _Transfer] = {}

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, GDSLoadStoreSpec)

        # Wait for GPU computation to finish before reading KV data
        torch.cuda.current_stream().synchronize()

        futures, num_bytes, file_handles = self._submit_writes(src_spec, dst_spec)

        self._transfers[job_id] = _Transfer(
            job_id=job_id,
            futures=futures,
            num_bytes=num_bytes,
            start_time=time.perf_counter(),
            file_handles=file_handles,
        )
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, GDSLoadStoreSpec)

        file_handles: list[tuple[CUfileHandle_t, int]] = []
        futures: list[Future] = []
        total_bytes = 0

        group_sizes = dst_spec.group_sizes
        block_indices = dst_spec.block_indices
        block_ids = dst_spec.block_ids

        key_idx = 0
        gpu_block_offset = 0
        for group_size, block_idx, group_data_refs in zip(
            group_sizes, block_indices, self._kv_cache_groups_data_refs
        ):
            if group_size == 0:
                continue
            n_offloaded = (
                group_size + self._block_size_factor - 1
            ) // self._block_size_factor
            logger.debug("submit_load: n_offloaded=%d", n_offloaded)
            for i in range(n_offloaded):
                key = src_spec.keys[key_idx]
                key_idx += 1
                start = gpu_block_offset + i * self._block_size_factor
                end = min(
                    start + self._block_size_factor,
                    gpu_block_offset + group_size,
                )
                gpu_blk_ids = block_ids[start:end]

                file_path = self._file_mapper.get_file_name(key)
                fd = open_for_gds(file_path, os.O_RDONLY)
                handle = cuFileHandleRegister(fd)
                file_handles.append((handle, fd))

                read_ops: list[tuple[int, int, int]] = []
                file_offset = 0
                for data_ref in group_data_refs:
                    t_idx = data_ref.tensor_idx
                    gpu_tensor = self._gpu_tensors[t_idx]
                    page_size = data_ref.page_size_bytes
                    base_ptr = gpu_tensor.data_ptr()
                    row_stride = gpu_tensor.stride(0)

                    for block_id in gpu_blk_ids:
                        dev_ptr = base_ptr + int(block_id) * row_stride
                        read_ops.append((dev_ptr, page_size, file_offset))
                        total_bytes += page_size
                        file_offset += page_size

                future = self._pool.submit(self._do_file_reads, handle, read_ops)
                futures.append(future)

            gpu_block_offset += group_size

        self._transfers[job_id] = _Transfer(
            job_id=job_id,
            futures=futures,
            num_bytes=total_bytes,
            start_time=time.perf_counter(),
            file_handles=file_handles,
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished_ids: list[int] = []
        for job_id, t in self._transfers.items():
            if all(f.done() for f in t.futures):
                # Propagate exceptions from threads
                for f in t.futures:
                    f.result()
                elapsed = time.perf_counter() - t.start_time
                results.append(
                    TransferResult(
                        job_id=t.job_id,
                        success=True,
                        transfer_size=t.num_bytes,
                        transfer_time=elapsed,
                    )
                )
                for handle, fd in t.file_handles:
                    cuFileHandleDeregister(handle)
                    os.close(fd)
                finished_ids.append(job_id)
        for job_id in finished_ids:
            del self._transfers[job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            t = self._transfers.get(job_id)
            if t is not None:
                for f in t.futures:
                    f.result()

    def shutdown(self) -> None:
        # Wait for all in-flight transfers
        for t in self._transfers.values():
            for f in t.futures:
                f.result()
            for handle, fd in t.file_handles:
                cuFileHandleDeregister(handle)
                os.close(fd)
        self._transfers.clear()

        self._pool.shutdown(wait=True)

        try:
            cuFileDriverClose()
        except RuntimeError as e:
            logger.warning("cuFileDriverClose failed: %s", e)
        logger.info("GDS worker shut down")

    # --- Internal: submit sync cuFile operations to thread pool ---

    def _submit_writes(
        self,
        src_spec: GPULoadStoreSpec,
        dst_spec: GDSLoadStoreSpec,
    ) -> tuple[list[Future], int, list[tuple[CUfileHandle_t, int]]]:
        """Submit cuFileWrite tasks to the thread pool, one per file."""
        file_handles: list[tuple[CUfileHandle_t, int]] = []
        futures: list[Future] = []
        total_bytes = 0

        group_sizes = src_spec.group_sizes
        block_indices = src_spec.block_indices
        block_ids = src_spec.block_ids

        key_idx = 0
        gpu_block_offset = 0
        for group_size, block_idx, group_data_refs in zip(
            group_sizes, block_indices, self._kv_cache_groups_data_refs
        ):
            if group_size == 0:
                continue
            n_offloaded = (
                group_size + self._block_size_factor - 1
            ) // self._block_size_factor
            for i in range(n_offloaded):
                key = dst_spec.keys[key_idx]
                key_idx += 1
                start = gpu_block_offset + i * self._block_size_factor
                end = min(
                    start + self._block_size_factor,
                    gpu_block_offset + group_size,
                )
                gpu_blk_ids = block_ids[start:end]

                file_path = self._file_mapper.get_file_name(key)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                fd = open_for_gds(file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
                handle = cuFileHandleRegister(fd)
                file_handles.append((handle, fd))

                write_ops: list[tuple[int, int, int]] = []
                file_offset = 0
                for data_ref in group_data_refs:
                    t_idx = data_ref.tensor_idx
                    gpu_tensor = self._gpu_tensors[t_idx]
                    page_size = data_ref.page_size_bytes
                    base_ptr = gpu_tensor.data_ptr()
                    row_stride = gpu_tensor.stride(0)

                    for block_id in gpu_blk_ids:
                        dev_ptr = base_ptr + int(block_id) * row_stride
                        write_ops.append((dev_ptr, page_size, file_offset))
                        total_bytes += page_size
                        file_offset += page_size

                future = self._pool.submit(self._do_file_writes, handle, write_ops)
                futures.append(future)

            gpu_block_offset += group_size
        return futures, total_bytes, file_handles

    @staticmethod
    def _do_file_writes(
        handle: CUfileHandle_t,
        ops: list[tuple[int, int, int]],
    ) -> None:
        """Execute sequential cuFileWrite calls for one file."""
        for dev_ptr, size, file_offset in ops:
            ret = cuFileWrite(handle, dev_ptr, size, file_offset)
            if ret != size:
                raise RuntimeError(f"cuFileWrite short write: {ret}/{size}")

    @staticmethod
    def _do_file_reads(
        handle: CUfileHandle_t,
        ops: list[tuple[int, int, int]],
    ) -> None:
        """Execute sequential cuFileRead calls for one file."""
        for dev_ptr, size, file_offset in ops:
            ret = cuFileRead(handle, dev_ptr, size, file_offset)
            if ret != size:
                raise RuntimeError(f"cuFileRead short read: {ret}/{size}")
