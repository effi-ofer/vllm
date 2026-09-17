# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading worker — direct GPU<->NVMe via cuFile sync API."""

import os
from concurrent.futures import Future

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
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

    All file syscalls (open, cuFileHandleRegister, I/O,
    cuFileHandleDeregister, close) run inside pool threads to keep the
    scheduler thread free of blocking operations.  Only cuFileDriverOpen
    and cuFileDriverClose run on the caller's thread at init/shutdown.
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

    def write_block(self, file_path: str, ops: list[tuple[int, int, int]]) -> None:
        logger.warning("write_block: %s (%d ops)", file_path, len(ops))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fd = open_for_gds(file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        try:
            handle = cuFileHandleRegister(fd)
            try:
                for dev_ptr, size, file_offset in ops:
                    ret = cuFileWrite(handle, dev_ptr, size, file_offset)
                    if ret != size:
                        raise RuntimeError(f"cuFileWrite short write: {ret}/{size}")
            finally:
                cuFileHandleDeregister(handle)
        finally:
            os.close(fd)

    def read_block(self, file_path: str, ops: list[tuple[int, int, int]]) -> None:
        logger.warning("read_block: %s (%d ops)", file_path, len(ops))
        fd = open_for_gds(file_path, os.O_RDONLY)
        try:
            handle = cuFileHandleRegister(fd)
            try:
                for dev_ptr, size, file_offset in ops:
                    ret = cuFileRead(handle, dev_ptr, size, file_offset)
                    if ret != size:
                        raise RuntimeError(f"cuFileRead short read: {ret}/{size}")
            finally:
                cuFileHandleDeregister(handle)
        finally:
            os.close(fd)

    def _submit_io(self, device_ptrs, keys, is_store):
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
                        prev_ptr, prev_size, prev_off = ops[-1] if ops else (0, 0, 0)
                        if ops and prev_ptr + prev_size == dev_ptr:
                            ops[-1] = (prev_ptr, prev_size + size, prev_off)
                        else:
                            ops.append((dev_ptr, size, file_offset))
                        file_offset += size
                        total_bytes += size

                file_path = self._file_mapper.get_file_name(key)
                if is_store:
                    future = self._pool.submit(self.write_block, file_path, ops)
                else:
                    future = self._pool.submit(self.read_block, file_path, ops)
                futures.append(future)

            ptr_offset += n_data_refs * group_size

        return futures, total_bytes

    def shutdown_backend(self) -> None:
        try:
            cuFileDriverClose()
        except RuntimeError as e:
            logger.warning("cuFileDriverClose failed: %s", e)
        logger.info("GDS worker shut down")
