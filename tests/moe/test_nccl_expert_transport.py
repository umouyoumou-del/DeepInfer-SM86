from __future__ import annotations

from deepinfer.moe.nccl_expert_transport import _pack_header, _unpack_header


def test_nccl_expert_header_roundtrip() -> None:
    payload = {
        "stop": False,
        "layer": 13,
        "experts": [3, 7, 19],
        "counts": [2, 1, 4],
        "shape": [3, 4, 2048],
        "dtype": "float16",
    }
    restored = _unpack_header(_pack_header(payload))
    assert restored == payload
    assert _unpack_header(_pack_header({"stop": True})) == {"stop": True}
