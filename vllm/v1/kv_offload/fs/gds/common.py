# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing_extensions import override

from vllm.v1.kv_offload.base import LoadStoreSpec, OffloadKey


class GDSOffloadingMetrics:
    GDS_STORES_TOTAL = "vllm:kv_offload_gds_stores_total"


class GDSLoadStoreSpec(LoadStoreSpec):
    """Spec carrying OffloadKeys that the worker resolves to file paths."""

    def __init__(self, keys: list[OffloadKey]):
        self.keys = keys

    @staticmethod
    @override
    def medium() -> str:
        return "GDS"
