# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
"""Device, memory and timing helpers: CPU-safe checks plus CUDA-only checks under the gpu marker."""

import pytest
import torch

from fluxlb.core import gpu

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


# --- device selection ---------------------------------------------------------


def test_get_device_returns_available_backend(monkeypatch):
    monkeypatch.delenv(gpu.DEVICE_ENV_VAR, raising=False)
    device = gpu.get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}
    if device.type == "cuda":
        assert torch.cuda.is_available()


def test_get_device_honours_env_override(monkeypatch):
    monkeypatch.setenv(gpu.DEVICE_ENV_VAR, "cpu")
    assert gpu.get_device() == torch.device("cpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the unavailable-CUDA path")
def test_get_device_rejects_unavailable_override(monkeypatch):
    monkeypatch.setenv(gpu.DEVICE_ENV_VAR, "cuda")
    with pytest.raises(RuntimeError, match=gpu.DEVICE_ENV_VAR):
        gpu.get_device()


def test_resolve_device_variants(monkeypatch):
    monkeypatch.setenv(gpu.DEVICE_ENV_VAR, "cpu")
    assert gpu.resolve_device(None) == torch.device("cpu")
    assert gpu.resolve_device("cpu") == torch.device("cpu")
    assert gpu.resolve_device(torch.device("cpu")) == torch.device("cpu")
    assert gpu.resolve_device(1) == torch.device("cuda", 1)


def test_accumulation_dtype_is_fp64_on_cuda_and_cpu():
    assert gpu.accumulation_dtype("cpu") is torch.float64
    assert gpu.accumulation_dtype(torch.device("cuda", 0)) is torch.float64


def test_accumulation_dtype_falls_back_on_mps_with_warning():
    with pytest.warns(UserWarning, match="float64"):
        assert gpu.accumulation_dtype(torch.device("mps")) is torch.float32


# --- GPU queries --------------------------------------------------------------


def test_gpu_queries_are_consistent():
    assert isinstance(gpu.is_gpu_available(), bool)
    assert gpu.get_gpu_count() >= 0
    assert (gpu.get_gpu_count() > 0) == gpu.is_gpu_available()
    info = gpu.get_gpu_info()
    status = gpu.poll_gpu_status()
    if gpu.is_gpu_available():
        assert info is not None and status is not None
        assert set(info) == set(status) == {f"gpu_{i}" for i in range(gpu.get_gpu_count())}
    else:
        assert info is None and status is None


def test_device_summary_mentions_torch_and_device(monkeypatch):
    monkeypatch.setenv(gpu.DEVICE_ENV_VAR, "cpu")
    summary = gpu.device_summary()
    assert torch.__version__ in summary
    assert "device=cpu" in summary


# --- memory -------------------------------------------------------------------


def test_format_bytes():
    assert gpu.format_bytes(0) == "0 B"
    assert gpu.format_bytes(512) == "512 B"
    assert gpu.format_bytes(2048) == "2.0 KiB"
    assert gpu.format_bytes(16 * 1024**3) == "16.0 GiB"


def test_memory_helpers_are_noops_on_cpu():
    assert gpu.memory_allocated("cpu") == 0
    assert gpu.peak_memory_allocated("cpu") == 0
    gpu.reset_peak_memory("cpu")
    gpu.clear_gpu_cache()


def test_track_memory_on_cpu_reports_untracked():
    with gpu.track_memory("cpu") as report:
        torch.zeros(16)
    assert report.tracked is False
    assert report.peak_allocated == 0
    assert report.delta_allocated == 0
    assert "not tracked" in str(report)


# --- timing -------------------------------------------------------------------


def test_synchronize_and_timer_on_cpu():
    gpu.synchronize("cpu")
    with gpu.Timer("cpu") as t:
        torch.ones(64) @ torch.ones(64)
    assert t.elapsed >= 0.0
    assert t.device == torch.device("cpu")


# --- CUDA only ----------------------------------------------------------------


@pytest.mark.gpu
@cuda_only
def test_gpu_info_fields():
    info = gpu.get_gpu_info()
    assert info is not None
    first = info["gpu_0"]
    assert set(first) == {
        "name",
        "total_memory",
        "current_memory_allocated",
        "current_memory_reserved",
    }
    assert first["total_memory"] > 0


@pytest.mark.gpu
@cuda_only
def test_track_memory_records_peak_on_cuda():
    gpu.clear_gpu_cache()
    with gpu.track_memory("cuda") as report:
        x = torch.zeros(1024, 1024, device="cuda")  # 4 MiB
        y = x + 1
        del x, y
    assert report.tracked is True
    assert report.peak_allocated >= 2 * 1024 * 1024 * 4
    assert report.delta_allocated == 0
    assert "peak allocated" in str(report)


@pytest.mark.gpu
@cuda_only
def test_device_summary_on_cuda():
    summary = gpu.device_summary("cuda")
    assert "device=cuda:" in summary
    assert "vram=" in summary
    assert "cc=" in summary
