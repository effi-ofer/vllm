# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import FSLoadStoreSpec, OffloadKey


class GDSOffloadingMetrics:
    GDS_STORES_TOTAL = "vllm:kv_offload_gds_stores_total"


class GDSLoadStoreSpec(FSLoadStoreSpec):
    """GDS-specific alias for FSLoadStoreSpec."""

    def __init__(self, keys: list[OffloadKey]):
        super().__init__(keys)

    @staticmethod
    def medium() -> str:
        return "GDS"
