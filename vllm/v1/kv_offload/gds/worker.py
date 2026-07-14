# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading worker — direct GPU<->NVMe via cuFile async API."""

import ctypes
import os
from collections import deque
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
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
    cuFileBufDeregister,
    cuFileBufRegister,
    cuFileDriverClose,
    cuFileDriverOpen,
    cuFileHandleDeregister,
    cuFileHandleRegister,
    cuFileReadAsync,
    cuFileStreamDeregister,
    cuFileStreamRegister,
    cuFileWriteAsync,
    open_for_gds,
)

logger = init_logger(__name__)


def _get_cuda_stream_ptr(stream: torch.cuda.Stream) -> int:
    """Get the raw CUstream pointer from a torch CUDA stream."""
    return stream.cuda_stream


@dataclass
class _Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    # Keep file handles alive until transfer completes
    file_handles: list[tuple[CUfileHandle_t, int]] = field(default_factory=list)
    # Keep ctypes arrays alive (async API reads them at execution time)
    pinned_params: list = field(default_factory=list)


class GDSOffloadingWorker(OffloadingWorker):
    """GPU<->NVMe transfers via NVIDIA cuFile async API.

    Uses CUDA streams for ordering — cuFileReadAsync/cuFileWriteAsync
    are enqueued on a stream and execute when the stream reaches them.
    Completion is tracked via CUDA events, matching the CPU offloading
    worker's design.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        file_mapper: FileMapper,
    ):
        self._file_mapper = file_mapper
        self._block_size_factor = block_size_factor
        self._kv_cache_groups_data_refs = kv_caches.group_data_refs

        # Initialize cuFile driver
        cuFileDriverOpen()
        logger.info("cuFile driver opened")

        # Register GPU KV cache buffers with cuFile
        self._registered_buffers: list[tuple[int, int]] = []
        self._gpu_tensors: list[torch.Tensor] = []

        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            self._gpu_tensors.append(gpu_tensor)

            ptr = gpu_tensor.data_ptr()
            size = gpu_tensor.numel() * gpu_tensor.element_size()
            cuFileBufRegister(ptr, size)
            self._registered_buffers.append((ptr, size))
            logger.debug(
                "Registered GPU buffer: ptr=0x%x size=%d (%.2f GB)",
                ptr,
                size,
                size / 1e9,
            )

        # Stream and event pools
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []
        self._registered_streams: set[int] = set()

        # In-flight transfers (FIFO order per direction)
        self._store_transfers: deque[_Transfer] = deque()
        self._load_transfers: deque[_Transfer] = deque()

    def _get_stream(self) -> torch.cuda.Stream:
        if self._stream_pool:
            return self._stream_pool.pop()
        stream = current_platform.Stream()
        stream_ptr = _get_cuda_stream_ptr(stream)
        cuFileStreamRegister(stream_ptr)
        self._registered_streams.add(stream_ptr)
        return stream

    def _get_event(self) -> torch.Event:
        if self._event_pool:
            return self._event_pool.pop()
        return torch.Event(enable_timing=True)

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, GDSLoadStoreSpec)

        stream = self._get_stream()
        start_event = self._get_event()
        end_event = self._get_event()

        # Wait for GPU computation to finish before reading KV data
        stream.wait_stream(current_platform.current_stream())
        # Serialize with previous transfer
        if self._store_transfers:
            stream.wait_event(self._store_transfers[-1].end_event)

        num_bytes, file_handles, pinned_params = self._enqueue_writes(
            stream, src_spec, dst_spec
        )

        with current_platform.stream(stream):
            start_event.record(stream)
            # Async writes already enqueued above
            end_event.record(stream)

        self._store_transfers.append(
            _Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_bytes,
                file_handles=file_handles,
                pinned_params=pinned_params,
            )
        )
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, GDSLoadStoreSpec)

        stream = self._get_stream()
        start_event = self._get_event()
        end_event = self._get_event()

        # Serialize with previous transfer
        if self._load_transfers:
            stream.wait_event(self._load_transfers[-1].end_event)

        num_bytes, file_handles, pinned_params = self._enqueue_reads(
            stream, src_spec, dst_spec
        )

        with current_platform.stream(stream):
            start_event.record(stream)
            end_event.record(stream)

        self._load_transfers.append(
            _Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_bytes,
                file_handles=file_handles,
                pinned_params=pinned_params,
            )
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for transfers in (self._store_transfers, self._load_transfers):
            while transfers and transfers[0].end_event.query():
                t = transfers.popleft()
                elapsed = t.start_event.elapsed_time(t.end_event) * 1e-3
                results.append(
                    TransferResult(
                        job_id=t.job_id,
                        success=True,
                        transfer_size=t.num_bytes,
                        transfer_time=elapsed,
                    )
                )
                # Close file handles now that transfer is done
                for handle, fd in t.file_handles:
                    cuFileHandleDeregister(handle)
                    os.close(fd)
                # Return stream/events to pool
                self._stream_pool.append(t.stream)
                self._event_pool.append(t.start_event)
                self._event_pool.append(t.end_event)
        return results

    def wait(self, job_ids: set[int]) -> None:
        for transfers in (self._store_transfers, self._load_transfers):
            for t in transfers:
                if t.job_id in job_ids:
                    t.end_event.synchronize()

    def shutdown(self) -> None:
        # Wait for all in-flight transfers
        for transfers in (self._store_transfers, self._load_transfers):
            while transfers:
                t = transfers.popleft()
                t.end_event.synchronize()
                for handle, fd in t.file_handles:
                    cuFileHandleDeregister(handle)
                    os.close(fd)

        # Deregister streams
        for stream_ptr in self._registered_streams:
            try:
                cuFileStreamDeregister(stream_ptr)
            except RuntimeError as e:
                logger.warning("cuFileStreamDeregister failed: %s", e)
        self._registered_streams.clear()
        self._stream_pool.clear()
        self._event_pool.clear()

        # Deregister GPU buffers
        for ptr, _size in self._registered_buffers:
            try:
                cuFileBufDeregister(ptr)
            except RuntimeError as e:
                logger.warning("cuFileBufDeregister failed: %s", e)
        self._registered_buffers.clear()

        try:
            cuFileDriverClose()
        except RuntimeError as e:
            logger.warning("cuFileDriverClose failed: %s", e)
        logger.info("GDS worker shut down")

    # --- Internal: enqueue async cuFile operations ---

    def _enqueue_writes(
        self,
        stream: torch.cuda.Stream,
        src_spec: GPULoadStoreSpec,
        dst_spec: GDSLoadStoreSpec,
    ) -> tuple[int, list[tuple[CUfileHandle_t, int]], list]:
        """Enqueue cuFileWriteAsync ops on the stream for each block."""
        file_handles: list[tuple[CUfileHandle_t, int]] = []
        pinned_params: list = []
        total_bytes = 0
        stream_ptr = _get_cuda_stream_ptr(stream)

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

                file_offset = 0
                for data_ref in group_data_refs:
                    t_idx = data_ref.tensor_idx
                    gpu_tensor = self._gpu_tensors[t_idx]
                    page_size = data_ref.page_size_bytes
                    base_ptr = gpu_tensor.data_ptr()
                    row_stride = gpu_tensor.stride(0)

                    for block_id in gpu_blk_ids:
                        buf_offset = int(block_id) * row_stride
                        params = self._make_async_params(
                            page_size, file_offset, buf_offset
                        )
                        pinned_params.append(params)
                        size_p, foff_p, boff_p, result_p = params

                        cuFileWriteAsync(
                            handle,
                            base_ptr,
                            size_p,
                            foff_p,
                            boff_p,
                            result_p,
                            stream_ptr,
                        )
                        total_bytes += page_size
                        file_offset += page_size

            gpu_block_offset += group_size
        return total_bytes, file_handles, pinned_params

    def _enqueue_reads(
        self,
        stream: torch.cuda.Stream,
        src_spec: GDSLoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> tuple[int, list[tuple[CUfileHandle_t, int]], list]:
        """Enqueue cuFileReadAsync ops on the stream for each block."""
        file_handles: list[tuple[CUfileHandle_t, int]] = []
        pinned_params: list = []
        total_bytes = 0
        stream_ptr = _get_cuda_stream_ptr(stream)

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

                file_offset = 0
                for data_ref in group_data_refs:
                    t_idx = data_ref.tensor_idx
                    gpu_tensor = self._gpu_tensors[t_idx]
                    page_size = data_ref.page_size_bytes
                    base_ptr = gpu_tensor.data_ptr()
                    row_stride = gpu_tensor.stride(0)

                    for block_id in gpu_blk_ids:
                        buf_offset = int(block_id) * row_stride
                        params = self._make_async_params(
                            page_size, file_offset, buf_offset
                        )
                        pinned_params.append(params)
                        size_p, foff_p, boff_p, result_p = params

                        cuFileReadAsync(
                            handle,
                            base_ptr,
                            size_p,
                            foff_p,
                            boff_p,
                            result_p,
                            stream_ptr,
                        )
                        total_bytes += page_size
                        file_offset += page_size

            gpu_block_offset += group_size
        return total_bytes, file_handles, pinned_params

    @staticmethod
    def _make_async_params(size: int, file_offset: int, buf_offset: int) -> tuple:
        """Create ctypes parameter arrays for cuFile async calls.

        The async API takes pointers that are read at stream execution
        time, so these must remain alive until the transfer completes.
        """
        size_arr = (ctypes.c_size_t * 1)(size)
        foff_arr = (ctypes.c_ssize_t * 1)(file_offset)
        boff_arr = (ctypes.c_ssize_t * 1)(buf_offset)
        result_arr = (ctypes.c_ssize_t * 1)(0)
        return size_arr, foff_arr, boff_arr, result_arr
