# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FSOffloadingWorker — ABC for file-based GPU offloading."""

import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    DevicePointers,
    OffloadingLoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.common import FSLoadStoreSpec

logger = init_logger(__name__)


@dataclass
class _Transfer:
    job_id: int
    futures: list[Future]
    num_bytes: int
    start_time: float
    context: list = field(default_factory=list)


class FSOffloadingWorker(OffloadingWorker, ABC):
    """ABC for file-based GPU offloading (GPU ↔ filesystem).

    Handles common orchestration: per-key pointer grouping, thread pool
    management, and completion tracking. Subclasses implement the actual
    I/O backend via write_block / read_block.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        block_size_factor: int = 1,
        max_io_threads: int = 32,
    ):
        self._file_mapper = file_mapper
        self._block_size_factor = block_size_factor
        self._pool = ThreadPoolExecutor(max_workers=max_io_threads)
        self._transfers: dict[int, _Transfer] = {}
        logger.info("FSOffloadingWorker thread pool: max_workers=%d", max_io_threads)

    # --- Abstract methods for subclasses ---

    @abstractmethod
    def write_block(self, file_path: str, ops: list[tuple[int, int, int]]) -> None:
        """Write GPU data to a file.

        Args:
            file_path: destination file path.
            ops: list of (dev_ptr, size_bytes, file_offset) tuples.
        """

    @abstractmethod
    def read_block(self, file_path: str, ops: list[tuple[int, int, int]]) -> None:
        """Read file data into GPU memory.

        Args:
            file_path: source file path.
            ops: list of (dev_ptr, size_bytes, file_offset) tuples.
        """

    # --- OffloadingWorker interface ---

    def submit_store(
        self,
        job_id: int,
        device_ptrs: DevicePointers,
        dst_spec: OffloadingLoadStoreSpec,
    ) -> bool:
        assert isinstance(dst_spec, FSLoadStoreSpec)

        futures: list[Future] = []
        total_bytes = 0
        key_idx = 0
        ptr_offset = 0

        for group_idx, (group_block_count, group_data_ref_count) in enumerate(
            zip(device_ptrs.group_block_counts, device_ptrs.group_data_ref_counts)
        ):
            group_total = group_block_count * group_data_ref_count
            if group_block_count == 0:
                ptr_offset += group_total
                continue

            n_offloaded = (
                group_block_count + self._block_size_factor - 1
            ) // self._block_size_factor

            for i in range(n_offloaded):
                key = dst_spec.keys[key_idx]
                key_idx += 1

                blk_start = i * self._block_size_factor
                blk_end = min(blk_start + self._block_size_factor, group_block_count)
                n_blocks_in_key = blk_end - blk_start

                file_path = self._file_mapper.get_file_name(key)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                write_ops: list[tuple[int, int, int]] = []
                file_offset = 0
                for d in range(group_data_ref_count):
                    base = ptr_offset + d * group_block_count + blk_start
                    for b in range(n_blocks_in_key):
                        idx = base + b
                        dev_ptr = int(device_ptrs.ptrs[idx])
                        size = int(device_ptrs.sizes[idx])
                        write_ops.append((dev_ptr, size, file_offset))
                        file_offset += size
                        total_bytes += size

                future = self._pool.submit(self.write_block, file_path, write_ops)
                futures.append(future)

            ptr_offset += group_total

        self._transfers[job_id] = _Transfer(
            job_id=job_id,
            futures=futures,
            num_bytes=total_bytes,
            start_time=time.perf_counter(),
        )
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: OffloadingLoadStoreSpec,
        device_ptrs: DevicePointers,
    ) -> bool:
        assert isinstance(src_spec, FSLoadStoreSpec)

        futures: list[Future] = []
        total_bytes = 0
        key_idx = 0
        ptr_offset = 0

        for group_idx, (group_block_count, group_data_ref_count) in enumerate(
            zip(device_ptrs.group_block_counts, device_ptrs.group_data_ref_counts)
        ):
            group_total = group_block_count * group_data_ref_count
            if group_block_count == 0:
                ptr_offset += group_total
                continue

            n_offloaded = (
                group_block_count + self._block_size_factor - 1
            ) // self._block_size_factor

            for i in range(n_offloaded):
                key = src_spec.keys[key_idx]
                key_idx += 1

                blk_start = i * self._block_size_factor
                blk_end = min(blk_start + self._block_size_factor, group_block_count)
                n_blocks_in_key = blk_end - blk_start

                file_path = self._file_mapper.get_file_name(key)

                read_ops: list[tuple[int, int, int]] = []
                file_offset = 0
                for d in range(group_data_ref_count):
                    base = ptr_offset + d * group_block_count + blk_start
                    for b in range(n_blocks_in_key):
                        idx = base + b
                        dev_ptr = int(device_ptrs.ptrs[idx])
                        size = int(device_ptrs.sizes[idx])
                        read_ops.append((dev_ptr, size, file_offset))
                        file_offset += size
                        total_bytes += size

                future = self._pool.submit(self.read_block, file_path, read_ops)
                futures.append(future)

            ptr_offset += group_total

        self._transfers[job_id] = _Transfer(
            job_id=job_id,
            futures=futures,
            num_bytes=total_bytes,
            start_time=time.perf_counter(),
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished_ids: list[int] = []
        for job_id, t in self._transfers.items():
            if all(f.done() for f in t.futures):
                for f in t.futures:
                    f.result()  # propagate exceptions
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

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            t = self._transfers.get(job_id)
            if t is not None:
                for f in t.futures:
                    f.result()

    def shutdown(self) -> None:
        for t in self._transfers.values():
            for f in t.futures:
                f.result()
        self._transfers.clear()
        self._pool.shutdown(wait=True)
        logger.info("FSOffloadingWorker shut down")
