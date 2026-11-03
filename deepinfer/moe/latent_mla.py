"""Latent MLA KV cache and attention for million-token contexts (DeepSeek-V2).

The per-head K/V that DeepSeek's attention builds are an up-projection of one
576-wide vector per token and layer: the normalized 512-d latent ``c`` and the
64-d rotary key ``k_pe``, both shared by all 16 heads.  Caching (c, k_pe)
instead of the expanded per-head K/V is what makes MLA small - 576 values per
token and layer instead of 16 x (192 + 128):

    expanded, fp16           324.0 KiB/token   (the paged pool's layout)
    latent, fp16              30.4 KiB/token
    latent fp8 + rope fp16    17.3 KiB/token   (1M tokens = 17.3 GiB)

Attention runs two ways over that cache:

* prefill (a chunk of queries): expand a tile of cached latents back to
  per-head K/V with ``kv_b_proj`` and run flash attention on it, merging tiles
  by log-sum-exp.  The expansion is T x 512 x 4096 MACs per tile, a few percent
  of the attention when the chunk holds thousands of queries.
* decode (one query): absorb W_UK into the query and W_UV into the output so
  the 16 heads attend directly over the shared latent, and every cached byte is
  read once per token for all heads.

Positions.  RoPE only touches the 64 rotary dims, and past the trained length
those rotations are out of distribution.  :class:`DualChunkPositions` is dual
chunk attention (An et al., 2024): keys are rotated by ``j mod chunk`` and a
query uses one of three rotations depending on whether the key is in its own
chunk, the previous one, or older.  Every token stays attendable and no
relative distance reaches the trained length; below the trained length it is
exactly ordinary RoPE.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

LATENT_DIM = 512
ROPE_DIM = 64
FP8_GROUP = 128
FP8_MAX = 448.0  # largest finite float8_e4m3fn


# ---------------------------------------------------------------- positions


@dataclass(frozen=True)
class DualChunkPositions:
    """Map absolute positions to the ones the rotary dims are allowed to see.

    ``trained`` is the length the model was trained to and ``chunk`` (<= it)
    the chunk size.  Keys sit at ``j mod chunk``.  A query at ``i`` uses

    * ``i mod chunk`` for keys in its own chunk (true distance),
    * ``min(i mod chunk + chunk, far)`` for the previous chunk (true distance,
      capped),
    * ``far`` for anything older, which puts those keys at distances in
      ``(far - chunk, far]``.

    ``far`` defaults to ``trained - 1``; a smaller one (>= ``chunk - 1``) keeps far keys away from
    the edge of the trained range, where long-range training data is thinnest
    (ChunkLlama caps near the chunk size).  It only changes queries - keys are
    stored rotated by ``j mod chunk`` - so it can differ between the prefill
    and the questions asked of a saved cache.  With ``chunk >= trained / 2``
    and ``i <= far`` all three agree with plain RoPE.
    ``DualChunkPositions.plain()`` is plain RoPE at any length.
    """

    trained: int
    chunk: int
    far: int = -1

    INTRA = 0
    SUCCESSIVE = 1
    INTER = 2

    def __post_init__(self) -> None:
        if self.chunk <= 0 or self.trained <= 0:
            raise ValueError("chunk and trained must be positive")
        if self.chunk > self.trained:
            raise ValueError("chunk must not exceed the trained length")
        if self.far < 0:
            object.__setattr__(self, "far", self.trained - 1)
        if not self.chunk - 1 <= self.far < self.trained:  # distances stay >= 0
            raise ValueError("far must lie in [chunk - 1, trained)")

    @classmethod
    def plain(cls) -> DualChunkPositions:
        big = 1 << 40
        return cls(trained=big, chunk=big)

    @classmethod
    def for_trained_length(cls, trained: int) -> DualChunkPositions:
        """The paper's default: chunks of three quarters of the trained length."""

        return cls(trained=int(trained), chunk=int(trained) * 3 // 4)

    def key_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return positions % self.chunk

    def query_positions(self, positions: torch.Tensor, relation: int) -> torch.Tensor:
        local = positions % self.chunk
        if relation == self.INTRA:
            return local
        if relation == self.SUCCESSIVE:
            return torch.clamp(local + self.chunk, max=self.far)
        if relation == self.INTER:
            return torch.full_like(positions, self.far)
        raise ValueError(f"unknown relation {relation}")

    def chunk_index(self, position: int) -> int:
        return int(position) // self.chunk

    def history_regions(self, start: int) -> list[tuple[int, int, int]]:
        """Key ranges ``[k0, k1)`` before ``start`` and how queries see them.

        Valid for any query block that starts at ``start`` and stays inside one
        chunk.  Returned oldest first.
        """

        current = self.chunk_index(start)
        own = current * self.chunk
        regions: list[tuple[int, int, int]] = []
        if current >= 2:
            regions.append((0, own - self.chunk, self.INTER))
        if current >= 1:
            regions.append((max(0, own - self.chunk), own, self.SUCCESSIVE))
        if start > own:
            regions.append((own, start, self.INTRA))
        return [r for r in regions if r[1] > r[0]]

    def split_block(self, start: int, length: int) -> list[tuple[int, int]]:
        """Cut ``[start, start+length)`` at chunk boundaries."""

        pieces = []
        position = int(start)
        end = int(start) + int(length)
        while position < end:
            boundary = (self.chunk_index(position) + 1) * self.chunk
            stop = min(end, boundary)
            pieces.append((position, stop - position))
            position = stop
        return pieces


# --------------------------------------------------------------------- rope


class LatentRope:
    """DeepSeek's rotary embedding, evaluated at arbitrary positions.

    The official module caches cos/sin up to the longest sequence seen, one
    copy per layer, with angles formed in fp32 - 13.8 GiB for 27 layers at 2M
    tokens, and an angle error of ~0.06 rad there.  This keeps one inverse
    frequency vector (YaRN already applied by the official module) and forms the
    angles in fp64 for just the positions asked for.
    """

    def __init__(self, inv_freq: torch.Tensor, *, mscale: float = 1.0) -> None:
        if inv_freq.numel() * 2 != ROPE_DIM:
            raise ValueError(f"expected {ROPE_DIM // 2} inverse frequencies")
        self.inv_freq = inv_freq.detach().to(torch.float64)
        self.mscale = float(mscale)

    @classmethod
    def from_attention(cls, attention, config=None) -> LatentRope:
        rotary = attention.rotary_emb
        mscale = 1.0
        scaling = getattr(config, "rope_scaling", None) if config is not None else None
        if scaling and str(scaling.get("type", "")).lower() == "yarn":
            factor = float(scaling["factor"])

            def get(scale: float, m: float) -> float:
                return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

            mscale = get(factor, float(scaling.get("mscale", 1))) / get(
                factor, float(scaling.get("mscale_all_dim", 0))
            )
        return cls(rotary.inv_freq, mscale=mscale)

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 ``[n, 64]`` tables in the official ``cat(freqs, freqs)`` layout."""

        device = positions.device
        angles = positions.to(torch.float64).unsqueeze(-1) * self.inv_freq.to(device)
        angles = torch.remainder(angles, 2 * math.pi)
        angles = torch.cat((angles, angles), dim=-1)
        return (
            (torch.cos(angles) * self.mscale).to(torch.float32),
            (torch.sin(angles) * self.mscale).to(torch.float32),
        )

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE exactly as ``modeling_deepseek.apply_rotary_pos_emb``.

        ``x`` is the raw projection ``[..., n, 64]`` with interleaved pairs; the
        official code de-interleaves before rotating, so the result is in the
        de-interleaved layout.  ``cos``/``sin`` are ``[n, 64]``.
        """

        shape = x.shape
        work = x.float().reshape(*shape[:-1], ROPE_DIM // 2, 2).transpose(-1, -2)
        work = work.reshape(shape)
        half = ROPE_DIM // 2
        rotated = torch.cat((-work[..., half:], work[..., :half]), dim=-1)
        return (work * cos + rotated * sin).to(x.dtype)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` of shape ``[n, 64]`` or ``[n, heads, 64]`` by ``positions [n]``."""

        cos, sin = self.cos_sin(positions)
        if x.dim() == 3:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return self.rotate(x, cos, sin)


# -------------------------------------------------------------------- fp8


def quantize_fp8_groups(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n, 512]`` -> fp8 e4m3 values and one fp32 scale per 128 values."""

    n, width = x.shape
    groups = x.float().reshape(n, width // FP8_GROUP, FP8_GROUP)
    scale = groups.abs().amax(dim=-1).clamp_min(1e-12) / FP8_MAX
    values = (groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return values.reshape(n, width), scale


def dequantize_fp8_groups(
    values: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    n, width = values.shape
    groups = values.float().reshape(n, width // FP8_GROUP, FP8_GROUP)
    return (groups * scale.unsqueeze(-1)).reshape(n, width).to(dtype)


# -------------------------------------------------------------------- cache


def pinned_empty(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """A page-locked host tensor of exactly ``shape``.

    ``torch.empty(..., pin_memory=True)`` rounds every request up to a power of
    two, so a 4M cache's 42 GiB latent array would pin 64 GiB - past the test
    host's 62 GiB container limit.  An ordinary allocation registered with CUDA
    pins only the bytes in use; it is unregistered before it is freed.
    """

    import weakref

    import numpy as np

    nbytes = math.prod(int(size) for size in shape) * torch.empty(0, dtype=dtype).element_size()
    if nbytes == 0:
        return torch.empty(tuple(shape), dtype=dtype)
    buffer = np.empty(nbytes, dtype=np.uint8)
    cudart = torch.cuda.cudart()
    pointer = buffer.ctypes.data
    error = cudart.cudaHostRegister(pointer, nbytes, 0)
    if error != cudart.cudaError.success:
        raise RuntimeError(f"could not pin {nbytes / 2**30:.2f} GiB of host memory: {error}")
    # runs when the last view of the buffer is gone, before numpy frees the memory
    weakref.finalize(buffer, cudart.cudaHostUnregister, pointer).atexit = False
    return torch.from_numpy(buffer).view(dtype).view(tuple(shape))


def drop_page_cache(path: str) -> None:
    """Let the kernel drop a file's cached pages; a container's memory limit counts them."""

    import os

    if not hasattr(os, "posix_fadvise"):
        return
    handle = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(handle)


class LatentKVCache:
    """Per-layer (latent, k_pe) storage for one long sequence.

    Positions ``[0, gpu_tokens)`` live on the GPU and the rest, if any, in
    pinned host memory.  ``kv_dtype`` is ``"fp16"`` or ``"fp8"``; fp8 stores the
    latent as e4m3 with a scale per 128 values and keeps k_pe in fp16, the same
    split FlashMLA uses, because the rotary dims carry position.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        capacity: int,
        gpu_tokens: int | None = None,
        kv_dtype: str = "fp16",
        device: torch.device | str = "cuda",
    ) -> None:
        if kv_dtype not in ("fp16", "fp8"):
            raise ValueError(f"kv_dtype must be fp16 or fp8, got {kv_dtype!r}")
        self.num_layers = int(num_layers)
        self.capacity = int(capacity)
        self.gpu_tokens = self.capacity if gpu_tokens is None else min(
            int(gpu_tokens), self.capacity
        )
        self.host_tokens = self.capacity - self.gpu_tokens
        self.kv_dtype = kv_dtype
        self.device = torch.device(device)
        self.length = 0
        latent_dtype = torch.float8_e4m3fn if kv_dtype == "fp8" else torch.float16

        def alloc(shape, dtype, *, host: bool) -> torch.Tensor:
            if host:
                return pinned_empty(shape, dtype)
            return torch.empty(shape, dtype=dtype, device=self.device)

        L, G, Hn = self.num_layers, self.gpu_tokens, self.host_tokens
        self.gpu_latent = alloc((L, G, LATENT_DIM), latent_dtype, host=False)
        self.gpu_pe = alloc((L, G, ROPE_DIM), torch.float16, host=False)
        self.gpu_scale = (
            alloc((L, G, LATENT_DIM // FP8_GROUP), torch.float32, host=False)
            if kv_dtype == "fp8"
            else None
        )
        if Hn > 0:
            self.host_latent = alloc((L, Hn, LATENT_DIM), latent_dtype, host=True)
            self.host_pe = alloc((L, Hn, ROPE_DIM), torch.float16, host=True)
            self.host_scale = (
                alloc((L, Hn, LATENT_DIM // FP8_GROUP), torch.float32, host=True)
                if kv_dtype == "fp8"
                else None
            )
        else:
            self.host_latent = self.host_pe = self.host_scale = None

    @staticmethod
    def bytes_per_token(num_layers: int, kv_dtype: str) -> int:
        latent = LATENT_DIM * (1 if kv_dtype == "fp8" else 2)
        scales = (LATENT_DIM // FP8_GROUP) * 4 if kv_dtype == "fp8" else 0
        return int(num_layers) * (latent + scales + ROPE_DIM * 2)

    def gpu_bytes(self) -> int:
        total = self.gpu_latent.numel() * self.gpu_latent.element_size()
        total += self.gpu_pe.numel() * self.gpu_pe.element_size()
        if self.gpu_scale is not None:
            total += self.gpu_scale.numel() * 4
        return int(total)

    def _segments(self, start: int, stop: int):
        """Split ``[start, stop)`` into the GPU part and the host part."""

        if start < self.gpu_tokens:
            yield "gpu", start, min(stop, self.gpu_tokens), start
        if stop > self.gpu_tokens:
            lo = max(start, self.gpu_tokens)
            yield "host", lo, stop, lo - self.gpu_tokens

    def write(
        self, layer: int, start: int, latent: torch.Tensor, pe: torch.Tensor
    ) -> None:
        """Store fp16 ``latent [n, 512]`` and rotated ``pe [n, 64]`` at ``start``."""

        n = latent.shape[0]
        stop = start + n
        if stop > self.capacity:
            raise ValueError(f"cache capacity {self.capacity} exceeded ({stop})")
        if self.kv_dtype == "fp8":
            values, scale = quantize_fp8_groups(latent)
        else:
            values, scale = latent.to(torch.float16), None
        pe = pe.to(torch.float16)
        for where, lo, hi, offset in self._segments(start, stop):
            a, b = lo - start, hi - start
            if where == "gpu":
                self.gpu_latent[layer, offset : offset + hi - lo].copy_(values[a:b])
                self.gpu_pe[layer, offset : offset + hi - lo].copy_(pe[a:b])
                if scale is not None:
                    self.gpu_scale[layer, offset : offset + hi - lo].copy_(scale[a:b])
            else:
                self.host_latent[layer, offset : offset + hi - lo].copy_(
                    values[a:b], non_blocking=True
                )
                self.host_pe[layer, offset : offset + hi - lo].copy_(
                    pe[a:b], non_blocking=True
                )
                if scale is not None:
                    self.host_scale[layer, offset : offset + hi - lo].copy_(
                        scale[a:b], non_blocking=True
                    )
        self.length = max(self.length, stop)

    def raw_tile(self, layer: int, start: int, stop: int):
        """Stored tensors for ``[start, stop)`` on the GPU (no dequantization).

        Returns ``(latent, scale_or_None, pe)``; a range that touches host
        memory is copied to the device.
        """

        parts = []
        for where, lo, hi, offset in self._segments(start, stop):
            count = hi - lo
            if where == "gpu":
                parts.append(
                    (
                        self.gpu_latent[layer, offset : offset + count],
                        None
                        if self.gpu_scale is None
                        else self.gpu_scale[layer, offset : offset + count],
                        self.gpu_pe[layer, offset : offset + count],
                    )
                )
            else:
                parts.append(
                    (
                        self.host_latent[layer, offset : offset + count].to(
                            self.device, non_blocking=True
                        ),
                        None
                        if self.host_scale is None
                        else self.host_scale[layer, offset : offset + count].to(
                            self.device, non_blocking=True
                        ),
                        self.host_pe[layer, offset : offset + count].to(
                            self.device, non_blocking=True
                        ),
                    )
                )
        if len(parts) == 1:
            return parts[0]
        # torch.cat has no float8 kernel; concatenate the raw bytes
        latent = torch.cat([p[0].view(torch.uint8) for p in parts]).view(parts[0][0].dtype)
        scale = None if parts[0][1] is None else torch.cat([p[1] for p in parts])
        pe = torch.cat([p[2] for p in parts])
        return latent, scale, pe

    def tile(self, layer: int, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
        """fp16 ``(latent [T, 512], pe [T, 64])`` for ``[start, stop)`` on the GPU."""

        latent, scale, pe = self.raw_tile(layer, start, stop)
        if scale is not None:
            latent = dequantize_fp8_groups(latent, scale)
        return latent, pe

    # ---------------------------------------------------------- persistence

    def _arrays(self):
        yield "latent", self.gpu_latent, self.host_latent
        yield "pe", self.gpu_pe, self.host_pe
        if self.gpu_scale is not None:
            yield "scale", self.gpu_scale, self.host_scale

    def _write_layer(self, handle, gpu: torch.Tensor, host, layer: int, count: int) -> None:
        for where, lo, hi, offset in self._segments(0, count):
            rows = (gpu if where == "gpu" else host)[layer, offset : offset + hi - lo]
            handle.write(rows.view(torch.uint8).cpu().numpy())

    def save(
        self,
        directory: str,
        *,
        meta: dict | None = None,
        spill: Sequence[str] = (),
        margin: int = 1 << 30,
    ) -> None:
        """Write the first ``length`` positions of every layer to ``directory``.

        Raw rows, layer-major, one file per array, plus ``cache.json``.
        Loading it back skips the prefill that produced it.

        With ``spill`` directories (other disks) every array is written as one
        file per layer instead, each into the first directory - ``directory``
        first - that keeps ``margin`` bytes free after it; ``cache.json`` records
        where each went.  A 2M fp8 cache is 34.6 GiB, more than either disk of
        the test host has.  Written pages are flushed and dropped from the page
        cache file by file, which a container's memory limit counts.
        """

        import json
        import os

        os.makedirs(directory, exist_ok=True)
        for folder in spill:
            os.makedirs(folder, exist_ok=True)
        directories = [directory, *spill]
        count = self.length
        files: dict[str, list[int]] = {}
        for name, gpu, host in self._arrays():
            if not spill:
                with open(os.path.join(directory, f"{name}.bin"), "wb") as handle:
                    for layer in range(self.num_layers):
                        self._write_layer(handle, gpu, host, layer, count)
                    _drop_pages(handle, written=True)
                continue
            size = count * gpu.shape[-1] * gpu.element_size()
            files[name] = []
            for layer in range(self.num_layers):
                index = next(
                    (i for i, folder in enumerate(directories) if _free_bytes(folder) >= size + margin), None
                )
                if index is None:
                    raise OSError(f"no room for {name} layer {layer} ({size / 2**30:.2f} GiB) in {directories}")
                with open(os.path.join(directories[index], f"{name}.{layer:02d}.bin"), "wb") as handle:
                    self._write_layer(handle, gpu, host, layer, count)
                    _drop_pages(handle, written=True)
                files[name].append(index)
        info = {
            "num_layers": self.num_layers,
            "length": count,
            "kv_dtype": self.kv_dtype,
            "meta": meta or {},
        }
        if spill:
            info["directories"] = [os.path.abspath(folder) for folder in directories]
            info["files"] = files
        with open(os.path.join(directory, "cache.json"), "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)

    @classmethod
    def load(
        cls,
        directory: str,
        *,
        capacity: int | None = None,
        gpu_tokens: int | None = None,
        device: torch.device | str = "cuda",
    ) -> tuple[LatentKVCache, dict]:
        """Inverse of :meth:`save`; returns the cache and the saved ``meta``.

        Host-memory rows are read straight into the (pinned) cache; GPU rows go
        through a bounded staging buffer.  A per-layer cache's first directory is
        ``directory`` itself, wherever it was saved.
        """

        import json
        import os

        with open(os.path.join(directory, "cache.json"), encoding="utf-8") as handle:
            info = json.load(handle)
        count = int(info["length"])
        cache = cls(
            num_layers=int(info["num_layers"]),
            capacity=max(int(capacity or 0), count),
            gpu_tokens=gpu_tokens,
            kv_dtype=info["kv_dtype"],
            device=device,
        )
        files = info.get("files")
        directories = info.get("directories") or [directory]
        for name, gpu, host in cache._arrays():
            row_bytes = gpu.shape[-1] * gpu.element_size()
            for layer in range(cache.num_layers):
                if files is None:
                    path, base = os.path.join(directory, f"{name}.bin"), layer * count * row_bytes
                else:
                    index = int(files[name][layer])
                    folder = directory if index == 0 else directories[index]
                    path, base = os.path.join(folder, f"{name}.{layer:02d}.bin"), 0
                with open(path, "rb", buffering=0) as handle:
                    for where, lo, hi, offset in cache._segments(0, count):
                        handle.seek(base + lo * row_bytes)
                        target = (gpu if where == "gpu" else host)[layer, offset : offset + hi - lo]
                        _read_rows(handle, target)
                    _drop_pages(handle, written=False)
        cache.length = count
        return cache, info.get("meta", {})


def _free_bytes(directory: str) -> int:
    import shutil

    return int(shutil.disk_usage(directory).free)


def _drop_pages(handle, *, written: bool) -> None:
    """Flush a file and let the kernel drop its pages (no-op where posix_fadvise is missing)."""

    import os

    if written:
        handle.flush()
        os.fsync(handle.fileno())
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _read_rows(handle, target: torch.Tensor, *, staging: int = 256 << 20) -> None:
    """Fill ``target`` (contiguous rows) from the file position: host rows in place, GPU rows via a staging buffer."""

    import numpy as np

    def read_exact(view: memoryview) -> None:
        while len(view):
            got = handle.readinto(view)
            if not got:
                raise OSError("cache file ended early")
            view = view[got:]

    if target.device.type == "cpu":
        read_exact(memoryview(target.view(torch.uint8).numpy()).cast("B"))
        return
    row_bytes = target.shape[-1] * target.element_size()
    step = max(1, staging // row_bytes)
    buffer = np.empty(min(target.shape[0], step) * row_bytes, dtype=np.uint8)
    for first in range(0, target.shape[0], step):
        rows = min(step, target.shape[0] - first)
        chunk = buffer[: rows * row_bytes]
        read_exact(memoryview(chunk))
        target[first : first + rows].copy_(torch.from_numpy(chunk).view(target.dtype).view(rows, -1))


# ---------------------------------------------------------------- attention


_FLASH: bool | None = None  # whether FlashAttention runs here; tried on first use


def _flash_runs(q: torch.Tensor) -> bool:
    probe = torch.zeros((1, 1, 8, q.shape[-1]), dtype=q.dtype, device=q.device)
    try:
        torch.ops.aten._scaled_dot_product_flash_attention(probe, probe, probe, 0.0, True, False, scale=1.0)
    except RuntimeError:  # not built in (PyTorch's Windows wheels) or not for this GPU
        return False
    return True


def flash_with_lse(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-2 returning ``(out [1,H,S,D] fp16, lse [1,H,S] fp32)``.

    Where FlashAttention is missing, the memory-efficient kernel gives the same
    pair (its log-sum-exp padded to a multiple of 32 rows).
    """

    global _FLASH
    if _FLASH is None:
        _FLASH = _flash_runs(q)
    if _FLASH:
        result = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, causal, False, scale=scale
        )
        return result[0], result[1]
    result = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0.0, causal, scale=scale
    )
    return result[0], result[1][..., : q.shape[-2]]


def merge_partials(
    out_a: torch.Tensor | None,
    lse_a: torch.Tensor | None,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two softmax-normalized partial attentions by their log-sum-exp."""

    out_b = out_b.float()
    if out_a is None:
        return out_b, lse_b
    top = torch.maximum(lse_a, lse_b)
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    wa = torch.exp(lse_a - top)
    wb = torch.exp(lse_b - top)
    total = wa + wb
    out = (out_a * wa.unsqueeze(-1) + out_b * wb.unsqueeze(-1)) / total.clamp_min(
        1e-30
    ).unsqueeze(-1)
    return out, top + torch.log(total.clamp_min(1e-30))


@dataclass
class MLAWeights:
    """The per-layer tensors the latent path needs, split per head."""

    num_heads: int
    nope_dim: int
    rope_dim: int
    v_dim: int
    softmax_scale: float
    kv_b: torch.Tensor  # [H*(nope+v), 512]
    w_uk: torch.Tensor  # [H, nope, 512]
    w_uv: torch.Tensor  # [H, v, 512]

    @classmethod
    def from_attention(cls, attention) -> MLAWeights:
        heads = int(attention.num_heads)
        nope = int(attention.qk_nope_head_dim)
        v_dim = int(attention.v_head_dim)
        kv_b = attention.kv_b_proj.weight
        per_head = kv_b.view(heads, nope + v_dim, kv_b.shape[-1])
        return cls(
            num_heads=heads,
            nope_dim=nope,
            rope_dim=int(attention.qk_rope_head_dim),
            v_dim=v_dim,
            softmax_scale=float(attention.softmax_scale),
            kv_b=kv_b,
            w_uk=per_head[:, :nope, :],
            w_uv=per_head[:, nope:, :],
        )

    def absorb_query(self, q_nope: torch.Tensor) -> torch.Tensor:
        """``q_nope [H, nope]`` -> ``q_abs [H, 512]`` fp16 (W_UK folded into the query)."""

        return torch.bmm(q_nope.to(self.w_uk.dtype).unsqueeze(1), self.w_uk).squeeze(1)

    def project_output(self, latent_out: torch.Tensor) -> torch.Tensor:
        """Attention-weighted latent ``[H, 512]`` -> ``[H, v_dim]`` fp16 (W_UV)."""

        weighted = latent_out.to(self.w_uv.dtype).unsqueeze(1)
        return torch.bmm(weighted, self.w_uv.transpose(1, 2)).squeeze(1)

    def expand(self, latent: torch.Tensor, pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent tile -> flash-ready ``k [1,H,T,192]`` and ``v [1,H,T,192]`` (v zero-padded)."""

        tokens = latent.shape[0]
        kv = torch.nn.functional.linear(latent, self.kv_b).view(
            tokens, self.num_heads, self.nope_dim + self.v_dim
        )
        qk_dim = self.nope_dim + self.rope_dim
        k = torch.empty(
            (1, self.num_heads, tokens, qk_dim), dtype=latent.dtype, device=latent.device
        )
        k[0, :, :, : self.nope_dim] = kv[:, :, : self.nope_dim].transpose(0, 1)
        k[0, :, :, self.nope_dim :] = pe.unsqueeze(0)
        v = torch.zeros(
            (1, self.num_heads, tokens, qk_dim), dtype=latent.dtype, device=latent.device
        )
        v[0, :, :, : self.v_dim] = kv[:, :, self.nope_dim :].transpose(0, 1)
        return k, v


def prefill_attention(
    weights: MLAWeights,
    cache: LatentKVCache,
    layer: int,
    q_nope: torch.Tensor,
    q_pe_raw: torch.Tensor,
    latent: torch.Tensor,
    pe: torch.Tensor,
    start: int,
    *,
    positions: DualChunkPositions,
    rope: LatentRope,
    tile_tokens: int = 32768,
    regions: list[tuple[int, int, int]] | None = None,
) -> torch.Tensor:
    """Attention for a query block at logical ``[start, start+n)`` inside one chunk.

    ``q_nope [n,H,128]`` and ``q_pe_raw [n,H,64]`` come from ``q_proj``;
    ``latent [n,512]`` and ``pe [n,64]`` (already rotated) are this block's own
    keys, used exactly for the causal diagonal.  History comes from ``cache``:
    ``regions`` lists ``(physical start, physical end, relation)`` for every
    earlier key; by default physical slot == logical position, ``[0, start)``.
    Returns ``[n, H, v_dim]`` fp16.
    """

    n = q_nope.shape[0]
    device = q_nope.device
    absolute = torch.arange(start, start + n, device=device)
    qk_dim = weights.nope_dim + weights.rope_dim
    queries: dict[int, torch.Tensor] = {}

    def query(relation: int) -> torch.Tensor:
        found = queries.get(relation)
        if found is None:
            rotated = rope.apply(
                q_pe_raw, positions.query_positions(absolute, relation)
            )
            found = torch.empty(
                (1, weights.num_heads, n, qk_dim), dtype=q_nope.dtype, device=device
            )
            found[0, :, :, : weights.nope_dim] = q_nope.transpose(0, 1)
            found[0, :, :, weights.nope_dim :] = rotated.transpose(0, 1)
            queries[relation] = found
        return found

    k, v = weights.expand(latent, pe)
    out, lse = flash_with_lse(
        query(positions.INTRA), k, v, causal=True, scale=weights.softmax_scale
    )
    acc, acc_lse = merge_partials(None, None, out[..., : weights.v_dim], lse)
    del k, v, out
    if regions is None:
        regions = positions.history_regions(start)
    for k0, k1, relation in regions:
        q = query(relation)
        for t0 in range(k0, k1, tile_tokens):
            t1 = min(t0 + tile_tokens, k1)
            tile_latent, tile_pe = cache.tile(layer, t0, t1)
            k, v = weights.expand(tile_latent, tile_pe)
            out, lse = flash_with_lse(q, k, v, causal=False, scale=weights.softmax_scale)
            acc, acc_lse = merge_partials(acc, acc_lse, out[..., : weights.v_dim], lse)
            del k, v, out, tile_latent, tile_pe
    return acc[0].transpose(0, 1).to(q_nope.dtype)


def extent_regions(
    extents: list[tuple[int, int, int]],
    query_position: int,
    positions: DualChunkPositions,
) -> list[tuple[int, int, int]]:
    """``(physical start, physical end, relation)`` for keys laid out by extents.

    An extent ``(p0, p1, l0)`` holds physical slots ``[p0, p1)`` at logical
    positions ``[l0, l0 + p1 - p0)``.  Each extent is cut where the logical
    position changes chunk, and every piece gets its relation to a query at
    logical ``query_position``.  Keys logically after the query's chunk are an
    error: the caller decides which extents a query may see.
    """

    query_chunk = positions.chunk_index(query_position)
    regions = []
    for p0, p1, l0 in extents:
        for start, length in positions.split_block(l0, p1 - p0):
            chunk = positions.chunk_index(start)
            if chunk == query_chunk:
                relation = positions.INTRA
            elif chunk == query_chunk - 1:
                relation = positions.SUCCESSIVE
            elif chunk < query_chunk - 1:
                relation = positions.INTER
            else:
                raise ValueError("a key lies in a later chunk than its query")
            physical = p0 + (start - l0)
            regions.append((physical, physical + length, relation))
    return regions


def _decode_regions(position: int, positions: DualChunkPositions):
    """Every key range a query at ``position`` attends to, itself included."""

    own = positions.chunk_index(position) * positions.chunk
    regions = [
        region
        for region in positions.history_regions(position)
        if region[2] != positions.INTRA
    ]
    regions.append((own, position + 1, positions.INTRA))
    return regions


def decode_attention(
    weights: MLAWeights,
    cache: LatentKVCache,
    layer: int,
    q_nope: torch.Tensor,
    q_pe_raw: torch.Tensor,
    position: int,
    *,
    positions: DualChunkPositions,
    rope: LatentRope,
    host_tile: int = 262144,
    regions: list[tuple[int, int, int]] | None = None,
) -> torch.Tensor:
    """Fused absorbed attention of one query; same contract as the reference.

    GPU-resident keys go straight to the Triton kernel; host-resident keys are
    copied over in tiles of ``host_tile`` and merged by log-sum-exp.
    """

    from deepinfer.moe import latent_mla_triton as fused

    if not fused.available():
        return decode_attention_reference(
            weights, cache, layer, q_nope, q_pe_raw, position,
            positions=positions, rope=rope, regions=regions,
        )
    device = q_nope.device
    q_abs = weights.absorb_query(q_nope)
    here = torch.tensor([position], device=device)
    if regions is None:
        regions = _decode_regions(position, positions)
    relations = sorted({relation for _, _, relation in regions})
    index_of = {relation: index for index, relation in enumerate(relations)}
    q_pe = torch.stack(
        [
            rope.apply(q_pe_raw.unsqueeze(0), positions.query_positions(here, relation))[0]
            for relation in relations
        ]
    ).to(torch.float16)
    gpu_segments, host_segments = [], []
    for k0, k1, relation in regions:
        for where, lo, hi, offset in cache._segments(k0, k1):
            target = gpu_segments if where == "gpu" else host_segments
            target.append((offset, offset + hi - lo, index_of[relation]))
    acc = lse = None
    if gpu_segments:
        out, out_lse = fused.decode_partial(
            q_abs,
            q_pe,
            cache.gpu_latent[layer],
            None if cache.gpu_scale is None else cache.gpu_scale[layer],
            cache.gpu_pe[layer],
            gpu_segments,
            sm_scale=weights.softmax_scale,
        )
        acc, lse = merge_partials(acc, lse, out, out_lse)
    for s0, s1, relation in host_segments:
        for t0 in range(s0, s1, host_tile):
            t1 = min(t0 + host_tile, s1)
            latent = cache.host_latent[layer, t0:t1].to(device, non_blocking=True)
            scale = (
                None
                if cache.host_scale is None
                else cache.host_scale[layer, t0:t1].to(device, non_blocking=True)
            )
            pe = cache.host_pe[layer, t0:t1].to(device, non_blocking=True)
            out, out_lse = fused.decode_partial(
                q_abs, q_pe, latent, scale, pe, [(0, t1 - t0, relation)],
                sm_scale=weights.softmax_scale,
            )
            acc, lse = merge_partials(acc, lse, out, out_lse)
    return weights.project_output(acc)


def decode_attention_reference(
    weights: MLAWeights,
    cache: LatentKVCache,
    layer: int,
    q_nope: torch.Tensor,
    q_pe_raw: torch.Tensor,
    position: int,
    *,
    positions: DualChunkPositions,
    rope: LatentRope,
    tile_tokens: int = 65536,
    regions: list[tuple[int, int, int]] | None = None,
) -> torch.Tensor:
    """Absorbed attention of one query at logical ``position``.

    The query's own (latent, pe) must already be in the cache.  ``regions``
    lists ``(physical start, physical end, relation)`` for every key including
    itself; by default the keys are physical slots ``[0, position]``.
    ``q_nope`` is ``[H, 128]``, ``q_pe_raw [H, 64]``; returns ``[H, v_dim]``
    fp16.  Plain PyTorch: the correctness reference for the fused kernel.
    """

    device = q_nope.device
    q_abs = torch.einsum("hd,hdk->hk", q_nope.float(), weights.w_uk.float())
    here = torch.tensor([position], device=device)
    if regions is None:
        regions = _decode_regions(position, positions)
    acc = acc_lse = None
    scale = weights.softmax_scale
    for k0, k1, relation in regions:
        q_pe = rope.apply(
            q_pe_raw.unsqueeze(0), positions.query_positions(here, relation)
        )[0].float()
        for t0 in range(k0, k1, tile_tokens):
            t1 = min(t0 + tile_tokens, k1)
            latent, pe = cache.tile(layer, t0, t1)
            latent = latent.float()
            scores = (q_abs @ latent.t() + q_pe @ pe.float().t()) * scale
            lse = torch.logsumexp(scores, dim=-1)
            partial = torch.exp(scores - lse.unsqueeze(-1)) @ latent
            acc, acc_lse = merge_partials(acc, acc_lse, partial, lse)
    out = torch.einsum("hk,hdk->hd", acc, weights.w_uv.float())
    return out.to(torch.float16)
