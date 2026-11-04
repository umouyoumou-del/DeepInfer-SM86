"""Runtime capability checks for Triton's native FP8 path.

Triton's FP8 E4M3 loads/dots require SM89 or newer.  The kernels in this
package therefore accept a ``SOFTWARE_FP8`` switch: older devices pass a uint8
view of the FP8 buffer and decode the E4M3 bytes inside the kernel.  Set
``DEEPINFER_TRITON_FP8=0`` to force the software path on a newer device (useful
for testing); ``=1`` forces the native path and is only valid on SM89+.
"""

from __future__ import annotations

import os

import torch


def native_fp8_available(device: torch.device | int | None = None) -> bool:
    """Whether Triton can load/dot FP8 E4M3 natively on ``device``."""

    setting = os.environ.get("DEEPINFER_TRITON_FP8", "auto").strip().lower()
    if setting in ("0", "false", "no", "off", "software"):
        return False
    if setting in ("1", "true", "yes", "on", "native"):
        return True
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


__all__ = ["native_fp8_available"]
