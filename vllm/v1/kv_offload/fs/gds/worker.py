# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading worker — direct GPU<->NVMe via cuFile sync API."""

import os

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import CanonicalKVCaches, DevicePointers
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.gds.cufile_bindings import (
    cuFileBufDeregister,
    cuFileBufRegister,
    cuFileDriverClose,
    cuFileDriverOpen,
    cuFileHandleDeregister,
    cuFileHandleRegister,
    cuFileRead,
    cuFileWrite,
    open_for_gds,
)
from vllm.v1.kv_offload.fs.worker import FSOffloadingWorker

logger = init_logger(__name__)

DEFAULT_MAX_THREADS = 400


class GDSOffloadingWorker(FSOffloadingWorker):
    """GPU<->NVMe transfers via NVIDIA cuFile synchronous API.

    Inherits orchestration (per-key grouping, thread pool, completion
    tracking) from FSOffloadingWorker. Implements write_block/read_block
    using cuFile for GPU-direct DMA to NVMe.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        file_mapper: FileMapper,
        max_io_threads: int = DEFAULT_MAX_THREADS,
    ):
        super().__init__(
            file_mapper=file_mapper,
            block_size_factor=block_size_factor,
            max_io_threads=max_io_threads,
        )

        # Initialize cuFile driver
        cuFileDriverOpen()
        logger.info("cuFile driver opened")

        # Register GPU buffers for DMA performance
        self._registered_bufs: list[int] = []
        for kv_cache_tensor in kv_caches.tensors:
            tensor = kv_cache_tensor.tensor.view(torch.int8)
            ptr = tensor.data_ptr()
            nbytes = tensor.numel() * tensor.element_size()
            try:
                cuFileBufRegister(ptr, nbytes, 0)
                self._registered_bufs.append(ptr)
            except RuntimeError as e:
                logger.warning(
                    "cuFileBufRegister failed for ptr=%#x size=%.1f MiB: %s "
                    "(will use unregistered path)",
                    ptr,
                    nbytes / (1 << 20),
                    e,
                )

        logger.info(
            "Registered %d/%d GPU buffers with cuFile",
            len(self._registered_bufs),
            len(kv_caches.tensors),
        )

    def write_block(self, file_path: str, ops: list[tuple[int, int, int]]) -> None:
        fd = open_for_gds(file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        handle = cuFileHandleRegister(fd)
        try:
            for dev_ptr, size, file_offset in ops:
                ret = cuFileWrite(handle, dev_ptr, size, file_offset)
                if ret != size:
                    raise RuntimeError(f"cuFileWrite short write: {ret}/{size}")
        finally:
            cuFileHandleDeregister(handle)
            os.close(fd)

    def read_block(self, file_path: str, ops: list[tuple[int, int, int]]) -> None:
        fd = open_for_gds(file_path, os.O_RDONLY)
        handle = cuFileHandleRegister(fd)
        try:
            for dev_ptr, size, file_offset in ops:
                ret = cuFileRead(handle, dev_ptr, size, file_offset)
                if ret != size:
                    raise RuntimeError(f"cuFileRead short read: {ret}/{size}")
        finally:
            cuFileHandleDeregister(handle)
            os.close(fd)

    def submit_store(self, job_id: int, device_ptrs: DevicePointers, dst_spec) -> bool:
        # Synchronize GPU before reading KV data
        torch.cuda.current_stream().synchronize()
        return super().submit_store(job_id, device_ptrs, dst_spec)

    def shutdown(self) -> None:
        super().shutdown()

        for ptr in self._registered_bufs:
            try:
                cuFileBufDeregister(ptr)
            except RuntimeError as e:
                logger.warning("cuFileBufDeregister failed: %s", e)
        self._registered_bufs.clear()

        try:
            cuFileDriverClose()
        except RuntimeError as e:
            logger.warning("cuFileDriverClose failed: %s", e)
        logger.info("GDS worker shut down")
