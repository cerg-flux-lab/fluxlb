# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Device selection, GPU memory and timing helpers.

The solver runs on three kinds of hardware: CUDA nodes such as mjolnir (16 GB, weak fp64),
Apple laptops through MPS (no fp64 at all), and plain CPU. Everything here exists so the rest
of the package can ask one place for the device, the accumulation dtype and memory or timing
figures, and behave identically on all three without sprinkling ``torch.cuda`` checks around.

Nothing in this module touches the collide-stream loop; it is plumbing only.
"""

from __future__ import annotations

import gc
import os
import platform
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

DEVICE_ENV_VAR = "FLUXLB_DEVICE"
"""Environment variable that overrides automatic device selection (e.g. ``cpu``, ``cuda:1``)."""

type DeviceLike = torch.device | str | int | None
"""Anything :func:`resolve_device` accepts: a device, a device string, a CUDA index or ``None``."""


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and bool(mps.is_available())


def get_device() -> torch.device:
    """Return the device to compute on.

    Resolution order is the ``FLUXLB_DEVICE`` environment variable if set, otherwise the first
    available of CUDA, MPS and CPU. The override lets a SLURM script or a test pin the device
    without editing code; an override that names an unavailable backend raises rather than
    silently falling back, so a job never quietly runs on the CPU.

    Returns
    -------
    torch.device
        The selected device.
    """
    override = os.environ.get(DEVICE_ENV_VAR)
    if override:
        device = torch.device(override)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"{DEVICE_ENV_VAR}={override!r} but CUDA is not available")
        if device.type == "mps" and not _mps_available():
            raise RuntimeError(f"{DEVICE_ENV_VAR}={override!r} but MPS is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Normalise a device-like argument to a :class:`torch.device`.

    ``None`` means :func:`get_device`, an ``int`` means that CUDA index, and a string is passed
    to :class:`torch.device`. Used by the per-device helpers below so callers can be loose.
    """
    if device is None:
        return get_device()
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def _cuda_index(device: DeviceLike) -> int | None:
    """Return the CUDA index for ``device``, or ``None`` if it is not a CUDA device."""
    dev = resolve_device(device)
    if dev.type != "cuda":
        return None
    return dev.index if dev.index is not None else torch.cuda.current_device()


def accumulation_dtype(device: DeviceLike = None) -> torch.dtype:
    """Return the dtype for accumulating conserved moments on ``device``.

    The project convention is fp32 compute with fp64 accumulation of density and momentum.
    MPS has no float64, so there the accumulation falls back to float32 and a warning is
    emitted once per call site, because mass and momentum conservation checks will be looser.

    Parameters
    ----------
    device
        Device to query; ``None`` selects :func:`get_device`.

    Returns
    -------
    torch.dtype
        ``torch.float64`` on CUDA and CPU, ``torch.float32`` on MPS.
    """
    dev = resolve_device(device)
    if dev.type == "mps":
        warnings.warn(
            "MPS does not support float64; accumulating conserved moments in float32. "
            "Expect looser mass and momentum conservation than on CUDA or CPU.",
            UserWarning,
            stacklevel=2,
        )
        return torch.float32
    return torch.float64


# ---------------------------------------------------------------------------
# GPU queries
# ---------------------------------------------------------------------------


def get_gpu_info() -> dict[str, dict[str, object]] | None:
    """Return information about the available GPU(s).

    If no GPU is available, it returns None.
    """
    if torch.cuda.is_available():
        return {
            f"gpu_{i}": {
                "name": torch.cuda.get_device_name(i),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
                "current_memory_allocated": torch.cuda.memory_allocated(i),
                "current_memory_reserved": torch.cuda.memory_reserved(i),
            }
            for i in range(torch.cuda.device_count())
        }
    else:
        return None


def get_gpu_count() -> int:
    """Return the number of available GPUs.

    If no GPU is available, it returns 0.
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def is_gpu_available() -> bool:
    """Return True if a GPU is available, otherwise return False."""
    return torch.cuda.is_available()


def poll_gpu_status() -> dict[str, dict[str, object]] | None:
    """Poll the status of the GPU(s).

    Returns a dictionary containing the current memory usage and other relevant information.
    If no GPU is available, it returns None.
    """
    if torch.cuda.is_available():
        gpu_status = {}
        for i in range(torch.cuda.device_count()):
            gpu_status[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
                "current_memory_allocated": torch.cuda.memory_allocated(i),
                "current_memory_reserved": torch.cuda.memory_reserved(i),
            }
        return gpu_status
    else:
        return None


def device_summary(device: DeviceLike = None) -> str:
    """Return a one-line description of the host and ``device`` for the top of a log.

    SLURM nodes are standalone, so every job log should say where it ran: hostname, torch and
    CUDA versions, and the GPU name, memory and compute capability. Print this at job start.
    """
    dev = resolve_device(device)
    parts = [
        f"host={platform.node()}",
        f"python={platform.python_version()}",
        f"torch={torch.__version__}",
    ]
    index = _cuda_index(dev)
    if index is not None:
        props = torch.cuda.get_device_properties(index)
        parts += [
            f"cuda={torch.version.cuda}",
            f"device=cuda:{index}",
            f"gpu={props.name!r}",
            f"vram={format_bytes(props.total_memory)}",
            f"cc={props.major}.{props.minor}",
            f"sms={props.multi_processor_count}",
        ]
    elif dev.type == "mps":
        parts += ["device=mps", f"machine={platform.machine()}"]
    else:
        parts += [
            "device=cpu",
            f"machine={platform.machine()}",
            f"threads={torch.get_num_threads()}",
        ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def format_bytes(n: int | float) -> str:
    """Format a byte count as a short human-readable string (e.g. ``'15.7 GiB'``)."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"  # pragma: no cover


def memory_allocated(device: DeviceLike = None) -> int:
    """Return bytes currently allocated by tensors on ``device`` (0 where not tracked)."""
    index = _cuda_index(device)
    return torch.cuda.memory_allocated(index) if index is not None else 0


def peak_memory_allocated(device: DeviceLike = None) -> int:
    """Return the peak bytes allocated on ``device`` since the last reset (0 where not tracked).

    This is the number to watch when tuning gradient checkpointing on a 16 GB card.
    """
    index = _cuda_index(device)
    return torch.cuda.max_memory_allocated(index) if index is not None else 0


def reset_peak_memory(device: DeviceLike = None) -> None:
    """Reset the peak-memory counters on ``device``; a no-op off CUDA."""
    index = _cuda_index(device)
    if index is not None:
        torch.cuda.reset_peak_memory_stats(index)


def clear_gpu_cache() -> None:
    """Release cached allocator memory so a following run or experiment starts clean.

    Garbage collection runs first so that tensors held only by cycles are actually freed;
    ``empty_cache`` cannot release memory that is still referenced from Python. Works on CUDA
    and MPS and is a no-op on CPU.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif _mps_available():
        torch.mps.empty_cache()


@dataclass
class MemoryReport:
    """Memory figures collected by :func:`track_memory` for one block of code.

    All values are bytes. On CPU and MPS the allocator is not tracked and every field is 0,
    with ``tracked`` set to ``False``.
    """

    device: torch.device
    tracked: bool
    allocated_before: int = 0
    allocated_after: int = 0
    peak_allocated: int = 0
    peak_reserved: int = 0

    @property
    def delta_allocated(self) -> int:
        """Bytes still allocated after the block that were not allocated before it."""
        return self.allocated_after - self.allocated_before

    def __str__(self) -> str:
        """Render the report as one human-readable line."""
        if not self.tracked:
            return f"[{self.device}] memory not tracked"
        return (
            f"[{self.device}] peak allocated {format_bytes(self.peak_allocated)}, "
            f"peak reserved {format_bytes(self.peak_reserved)}, "
            f"net change {format_bytes(self.delta_allocated)}"
        )


@contextmanager
def track_memory(device: DeviceLike = None) -> Iterator[MemoryReport]:
    """Track peak and net memory use of a block on ``device``.

    The peak counters are reset on entry and read on exit after synchronising, so the report
    reflects only the enclosed block. Use it around one time step, one training step or one
    rollout to size checkpointing and batch dimensions against the available VRAM.

    Examples
    --------
    >>> with track_memory() as report:
    ...     f = solver.run(f, n_steps=10)
    >>> print(report)
    """
    dev = resolve_device(device)
    index = _cuda_index(dev)
    report = MemoryReport(device=dev, tracked=index is not None)
    if index is not None:
        synchronize(dev)
        torch.cuda.reset_peak_memory_stats(index)
        report.allocated_before = torch.cuda.memory_allocated(index)
    try:
        yield report
    finally:
        if index is not None:
            synchronize(dev)
            report.allocated_after = torch.cuda.memory_allocated(index)
            report.peak_allocated = torch.cuda.max_memory_allocated(index)
            report.peak_reserved = torch.cuda.max_memory_reserved(index)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def synchronize(device: DeviceLike = None) -> None:
    """Block until all queued work on ``device`` has finished; a no-op on CPU.

    GPU kernels are launched asynchronously, so any wall-clock measurement or memory reading
    taken without synchronising first describes the launch queue, not the computation.
    """
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Context manager that measures wall-clock time of a block with device synchronisation.

    Synchronises ``device`` on entry and exit so the interval covers the GPU work actually
    done, not merely the kernel launches. ``elapsed`` is in seconds and is valid after exit.

    Examples
    --------
    >>> with Timer() as t:
    ...     f = solver.run(f, n_steps=100)
    >>> print(f"{t.elapsed:.3f} s")
    """

    def __init__(self, device: DeviceLike = None) -> None:
        self.device = resolve_device(device)
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        """Synchronise and start the clock."""
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        """Synchronise and stop the clock, storing ``elapsed``."""
        synchronize(self.device)
        self.elapsed = time.perf_counter() - self._start
