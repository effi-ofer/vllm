# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading worker — direct GPU<->NVMe via cuFile sync API."""

import os

from vllm.logger import init_logger
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.worker import DEFAULT_MAX_THREADS, FSOffloadingWorker
from vllm.v1.kv_offload.gds.cufile_bindings import (
    cuFileDriverClose,
    cuFileDriverOpen,
    cuFileHandleDeregister,
    cuFileHandleRegister,
    cuFileRead,
    cuFileWrite,
    open_for_gds,
)

logger = init_logger(__name__)


class GDSOffloadingWorker(FSOffloadingWorker):
    """GPU<->NVMe transfers via NVIDIA cuFile synchronous API.

    Each file's I/O is submitted to a thread pool so multiple operations
    overlap on the NVMe device (higher queue depth = higher throughput).

    cuFileDriverOpen, cuFileBufRegister, and cuFileHandleRegister all run
    on the calling (main) thread which holds the RDMA context.  Only the
    actual cuFileWrite/cuFileRead calls run in the pool (which needs CUDA
    context for GPU memory access).

    Buffer registration is per-request: only the blocks involved in the
    current I/O are registered, keeping BAR usage minimal.
    """

    def __init__(
        self,
        block_size_factor: int,
        file_mapper: FileMapper,
        max_io_threads: int = DEFAULT_MAX_THREADS,
    ):
        super().__init__(
            file_mapper=file_mapper,
            block_size_factor=block_size_factor,
            max_io_threads=max_io_threads,
        )

        cuFileDriverOpen()
        logger.info("cuFile driver opened")
        self._job_file_handles: dict[int, list[tuple]] = {}

    def write_block(
        self, file_path: str, ops: list[tuple[int, int, int]]
    ) -> object | None:
        raise NotImplementedError

    def read_block(
        self, file_path: str, ops: list[tuple[int, int, int]]
    ) -> object | None:
        raise NotImplementedError

    @staticmethod
    def _do_writes(handle, ops):
        for dev_ptr, size, file_offset in ops:
            ret = cuFileWrite(handle, dev_ptr, size, file_offset)
            if ret != size:
                raise RuntimeError(f"cuFileWrite short write: {ret}/{size}")

    @staticmethod
    def _do_reads(handle, ops):
        for dev_ptr, size, file_offset in ops:
            ret = cuFileRead(handle, dev_ptr, size, file_offset)
            if ret != size:
                raise RuntimeError(f"cuFileRead short read: {ret}/{size}")

    def _submit_io(self, device_ptrs, keys, is_store):
        from concurrent.futures import Future

        from vllm.utils.math_utils import cdiv

        futures: list[Future] = []
        total_bytes = 0

        group_block_counts = device_ptrs.group_block_counts
        group_data_ref_counts = device_ptrs.group_data_ref_counts

        key_idx = 0
        ptr_offset = 0

        for group_size, n_data_refs in zip(group_block_counts, group_data_ref_counts):
            if group_size == 0:
                continue

            n_files = cdiv(group_size, self._block_size_factor)
            for i in range(n_files):
                key = keys[key_idx]
                key_idx += 1

                blk_start = i * self._block_size_factor
                blk_end = min(blk_start + self._block_size_factor, group_size)
                n_blks = blk_end - blk_start

                ops: list[tuple[int, int, int]] = []
                file_offset = 0
                for d in range(n_data_refs):
                    base = ptr_offset + d * group_size + blk_start
                    for b in range(n_blks):
                        idx = base + b
                        dev_ptr = int(device_ptrs.ptrs[idx])
                        size = int(device_ptrs.sizes[idx])
                        ops.append((dev_ptr, size, file_offset))
                        file_offset += size
                        total_bytes += size

                file_path = self._file_mapper.get_file_name(key)
                if is_store:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    fd = open_for_gds(file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
                    handle = cuFileHandleRegister(fd)
                    future = self._pool.submit(self._do_writes, handle, ops)
                else:
                    fd = open_for_gds(file_path, os.O_RDONLY)
                    handle = cuFileHandleRegister(fd)
                    future = self._pool.submit(self._do_reads, handle, ops)
                self._file_handles.append((handle, fd))
                futures.append(future)

            ptr_offset += n_data_refs * group_size

        return futures, total_bytes

    def submit_store(self, job_id, device_ptrs, dst_spec):
        self._file_handles: list[tuple] = []
        result = super().submit_store(job_id, device_ptrs, dst_spec)
        self._job_file_handles[job_id] = self._file_handles
        return result

    def submit_load(self, job_id, src_spec, device_ptrs):
        self._file_handles = []
        result = super().submit_load(job_id, src_spec, device_ptrs)
        self._job_file_handles[job_id] = self._file_handles
        return result

    def get_finished(self):
        import time

        from vllm.v1.kv_offload.base import TransferResult

        results: list[TransferResult] = []
        finished_ids: list[int] = []
        for job_id, t in self._transfers.items():
            if all(f.done() for f in t.futures):
                for f in t.futures:
                    f.result()
                for handle, fd in self._job_file_handles.pop(job_id, []):
                    cuFileHandleDeregister(handle)
                    os.close(fd)
                elapsed = time.perf_counter() - t.start_time
                results.append(
                    TransferResult(
                        job_id=t.job_id,
                        success=True,
                        transfer_size=t.num_bytes,
                        transfer_time=elapsed,
                    )
                )
                finished_ids.append(job_id)
        for job_id in finished_ids:
            del self._transfers[job_id]
        return results

    def shutdown_backend(self) -> None:
        try:
            cuFileDriverClose()
        except RuntimeError as e:
            logger.warning("cuFileDriverClose failed: %s", e)
        logger.info("GDS worker shut down")
