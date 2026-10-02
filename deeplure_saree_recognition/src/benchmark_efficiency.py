"""
Efficiency Benchmarking Suite: Parameter Count, FLOPs/MACs, Latency & Throughput.
Author: Computer Vision Engineering Team
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Returns:
      total_params: Total number of parameters
      trainable_params: Number of trainable parameters
      model_size_mb: Memory footprint of parameters in Megabytes (FP32)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4.0) / (1024 * 1024)
    return total_params, trainable_params, model_size_mb


def estimate_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> float:
    """
    Estimates Multiply-Accumulate Operations (MACs/FLOPs) using forward hooks.
    """
    macs = 0

    def conv_hook(module, input, output):
        nonlocal macs
        # Output spatial dims
        _, _, out_h, out_w = output.shape
        # kernel_h * kernel_w * (in_channels / groups) * out_channels * out_h * out_w
        kernel_ops = module.weight.shape[2] * module.weight.shape[3] * (module.in_channels // module.groups)
        macs += kernel_ops * module.out_channels * out_h * out_w

    def linear_hook(module, input, output):
        nonlocal macs
        macs += module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)
    
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    gflops = (macs * 2.0) / 1e9  # 1 MAC ~ 2 FLOPs
    return gflops


def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    warmup_runs: int = 30,
    timed_runs: int = 100
) -> Dict[str, float]:
    """
    Measures inference latency and throughput under realistic deployment conditions.
    Uses CUDA events for exact GPU hardware timestamping when running on CUDA.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size, device=device)

    # Warm-up cycles to trigger PyTorch JIT and GPU power scaling
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    latencies_ms = []

    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for _ in range(timed_runs):
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                latencies_ms.append(start_event.elapsed_time(end_event))
    else:
        with torch.no_grad():
            for _ in range(timed_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    mean_latency = float(np.mean(latencies_arr))
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    return {
        "device": device,
        "mean_latency_ms": round(mean_latency, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "throughput_fps": round(fps, 1)
    }


def run_full_efficiency_benchmark(model: nn.Module) -> Dict[str, Any]:
    """Executes a full efficiency audit across parameters, FLOPs, and latency."""
    total_params, trainable_params, model_size_mb = count_parameters(model)
    gflops = estimate_flops(model)
    
    # Latency on active device
    active_device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_stats = measure_latency(model, device=active_device)
    
    # Latency on CPU as baseline
    cpu_stats = measure_latency(model, device="cpu", timed_runs=30)
    
    # Embedding memory footprint (e.g. 256-D float32)
    embedding_dim = getattr(model, "embedding_dim", 256)
    bytes_per_embedding_fp32 = embedding_dim * 4
    bytes_per_embedding_fp16 = embedding_dim * 2
    mb_per_100k_sarees_fp16 = (100_000 * bytes_per_embedding_fp16) / (1024 * 1024)

    return {
        "total_parameters_M": round(total_params / 1e6, 2),
        "trainable_parameters_M": round(trainable_params / 1e6, 2),
        "model_weight_size_MB": round(model_size_mb, 2),
        "GFLOPs": round(gflops, 3),
        "embedding_dim": embedding_dim,
        "embedding_size_bytes_fp32": bytes_per_embedding_fp32,
        "embedding_size_bytes_fp16": bytes_per_embedding_fp16,
        "catalog_100k_index_size_MB_fp16": round(mb_per_100k_sarees_fp16, 2),
        "active_device_latency": gpu_stats,
        "cpu_latency": cpu_stats
    }
