# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading manager — unlimited capacity, no eviction."""

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.gds.common import GDSLoadStoreSpec


@dataclass
class _BlockState:
    is_ready: bool = False
    ref_cnt: int = 0


class GDSOffloadingManager(OffloadingManager):
    """Tracks offloaded blocks on GDS storage.

    Unlike CPUOffloadingManager, there is no fixed capacity and no
    eviction.  Blocks persist on disk until reset_cache() or shutdown.
    """

    def __init__(self, enable_events: bool = False):
        self._blocks: dict[OffloadKey, _BlockState] = {}
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        state = self._blocks.get(key)
        if state is None:
            return LookupResult.MISS
        if not state.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        for key in keys:
            state = self._blocks.get(key)
            assert state is not None, f"Block {key!r} not found"
            assert state.is_ready, f"Block {key!r} not ready"
            state.ref_cnt += 1
        return GDSLoadStoreSpec(list(keys))

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        pass

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            state = self._blocks.get(key)
            assert state is not None, f"Block {key!r} not found"
            assert state.ref_cnt > 0
            state.ref_cnt -= 1

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        keys_to_store = [k for k in keys if k not in self._blocks]
        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=GDSLoadStoreSpec([]),
                evicted_keys=[],
            )
        for key in keys_to_store:
            self._blocks[key] = _BlockState(is_ready=False)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=GDSLoadStoreSpec(keys_to_store),
            evicted_keys=[],
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []
        if success:
            for key in keys:
                state = self._blocks.get(key)
                if state is not None and not state.is_ready:
                    state.is_ready = True
                    stored_keys.append(key)
        else:
            for key in keys:
                state = self._blocks.get(key)
                if state is not None and not state.is_ready:
                    del self._blocks[key]

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=GDSLoadStoreSpec.medium(),
                    removed=False,
                )
            )

    @override
    def reset_cache(self) -> None:
        self._blocks.clear()

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
