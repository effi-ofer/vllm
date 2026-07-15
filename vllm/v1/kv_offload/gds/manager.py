# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading manager — unlimited capacity, no eviction."""

import os
import sys
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.gds.common import GDSLoadStoreSpec

logger = init_logger(__name__)


class GDSOffloadingManager(CPUOffloadingManager):
    """Tracks offloaded blocks on GDS storage.

    Inherits ref-counting and load/store lifecycle from
    CPUOffloadingManager.  Overrides capacity to be unlimited (no
    eviction) and produces GDSLoadStoreSpec (keys) instead of
    CPULoadStoreSpec (block IDs).

    On lookup miss, falls back to a filesystem check so blocks
    persisted by previous sessions are discovered.
    """

    def __init__(self, file_mapper: FileMapper, enable_events: bool = False):
        super().__init__(
            num_blocks=sys.maxsize,
            cache_policy="lru",
            enable_events=enable_events,
            store_threshold=0,
        )
        self.medium = GDSLoadStoreSpec.medium()
        self._file_mapper = file_mapper

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = super().lookup(key, req_context)
        if result != LookupResult.MISS:
            return result
        # Check if block exists on disk from a previous session
        file_path = self._file_mapper.get_file_name(key)
        if os.path.exists(file_path):
            blocks = self._allocate_blocks([key])
            block = blocks[0]
            block.ref_cnt = 0
            self._policy.insert(key, block)
            self._num_evictable_cache_blocks += 1
            self._policy.mark_evictable(key)
            return LookupResult.HIT
        return LookupResult.MISS

    @override
    def _get_num_free_blocks(self) -> int:
        return sys.maxsize

    @override
    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        blocks: list[BlockStatus] = []
        for _ in keys:
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1
        return blocks

    @override
    def _get_load_store_spec(  # type: ignore[override]
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> LoadStoreSpec:
        return GDSLoadStoreSpec(list(keys))

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        logger.debug(
            "prepare_load: %d keys, req=%s",
            len(list(keys)),
            req_context.req_id,
        )
        return super().prepare_load(keys, req_context)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        keys_to_store = [k for k in keys if self._policy.get(k) is None]
        if not keys_to_store:
            logger.debug(
                "prepare_store: %d keys (0 new, 0 blocks), req=%s",
                len(list(keys)),
                req_context.req_id,
            )
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=GDSLoadStoreSpec([]),
                evicted_keys=[],
            )
        blocks = self._allocate_blocks(keys_to_store)
        logger.debug(
            "prepare_store: %d keys (%d new, %d blocks), req=%s",
            len(list(keys)),
            len(keys_to_store),
            len(blocks),
            req_context.req_id,
        )
        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        store_spec = self._get_load_store_spec(keys_to_store, blocks)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=[],
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        logger.debug(
            "complete_store: %d keys, success=%s, req=%s",
            len(list(keys)),
            success,
            req_context.req_id,
        )
        super().complete_store(keys, req_context, success)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        logger.debug(
            "complete_load: %d keys, req=%s",
            len(list(keys)),
            req_context.req_id,
        )
        super().complete_load(keys, req_context)
