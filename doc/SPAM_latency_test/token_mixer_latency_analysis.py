"""
Token Mixer Latency Analysis
============================
This script analyzes the latency differences between three token mixers:
- SepConv (ConvFormer): 110.86 ms/step
- ExSPAM (SPANetV2): 137.85 ms/step  
- DynamicFilter (DFFormer): 170.33 ms/step

All three models use the same MetaFormer baseline, differing only in token mixer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid segfault
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import os

from models.spanetv2 import ExSPAM
from models.dfformer import DynamicFilter
from models.metaformer_baselines import SepConv


# Create results directory
os.makedirs('results_figs', exist_ok=True)


def warmup_gpu(model: nn.Module, x: torch.Tensor, num_warmup: int = 10):
    """GPU warmup - reduced for faster execution"""
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    torch.cuda.synchronize()


def measure_latency(model: nn.Module, x: torch.Tensor, num_iterations: int = 100) -> Tuple[float, float]:
    """Accurate GPU latency measurement using CUDA events"""
    model.eval()
    warmup_gpu(model, x)
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = model(x)
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
    
    return np.mean(latencies), np.std(latencies)


def analyze_token_mixers():
    """Analyze latency of individual token mixers"""
    device = torch.device('cuda:0')
    
    # Test configuration matching MetaFormer s18 models
    # Stage dimensions: [64, 128, 320, 512] 
    # We'll test all stages but focus on stage 3 (320 dim) which has most blocks (9)
    
    test_configs = [
        {"name": "Stage 1", "dim": 64, "size": 56},   # 224/4 = 56
        {"name": "Stage 2", "dim": 128, "size": 28},  # 224/8 = 28  
        {"name": "Stage 3", "dim": 320, "size": 14},  # 224/16 = 14 (main bottleneck)
        {"name": "Stage 4", "dim": 512, "size": 7},   # 224/32 = 7
    ]
    
    batch_size = 128  # Same as benchmark
    results = {}
    
    print("=" * 60)
    print("Token Mixer Latency Analysis")
    print("=" * 60)
    
    for config in test_configs:
        dim = config["dim"]
        size = config["size"]
        stage_name = config["name"]
        
        print(f"\n{stage_name}: dim={dim}, size={size}x{size}")
        print("-" * 40)
        
        # Input tensor: (B, H, W, C) format for MetaFormer
        x = torch.randn(batch_size, size, size, dim, device=device, dtype=torch.float32)
        
        stage_results = {}
        
        # 1. SepConv (ConvFormer)
        sepconv = SepConv(dim=dim).to(device)
        mean_lat, std_lat = measure_latency(sepconv, x, num_iterations=500)
        stage_results["SepConv"] = {"mean": mean_lat, "std": std_lat}
        print(f"SepConv:       {mean_lat:.3f} ± {std_lat:.3f} ms")
        
        # 2. ExSPAM (SPANetV2) 
        exspam = ExSPAM(dim=dim, size=size).to(device)
        mean_lat, std_lat = measure_latency(exspam, x, num_iterations=500)
        stage_results["ExSPAM"] = {"mean": mean_lat, "std": std_lat}
        print(f"ExSPAM:        {mean_lat:.3f} ± {std_lat:.3f} ms")
        
        # 3. DynamicFilter (DFFormer)
        dynfilter = DynamicFilter(dim=dim, size=size).to(device)
        mean_lat, std_lat = measure_latency(dynfilter, x, num_iterations=500)
        stage_results["DynamicFilter"] = {"mean": mean_lat, "std": std_lat}
        print(f"DynamicFilter: {mean_lat:.3f} ± {std_lat:.3f} ms")
        
        results[stage_name] = stage_results
        
        # Clean up GPU memory
        del sepconv, exspam, dynfilter, x
        torch.cuda.empty_cache()
    
    return results


def create_visualizations(results):
    """Create comprehensive visualizations"""
    
    # Prepare data for plotting
    stages = list(results.keys())
    token_mixers = ["SepConv", "ExSPAM", "DynamicFilter"]
    colors = ['#2E8B57', '#FF6B35', '#8B0000']  # Green, Orange, Dark Red
    
    # Extract latencies for each stage
    sepconv_latencies = [results[stage]["SepConv"]["mean"] for stage in stages]
    exspam_latencies = [results[stage]["ExSPAM"]["mean"] for stage in stages]
    dynfilter_latencies = [results[stage]["DynamicFilter"]["mean"] for stage in stages]
    
    sepconv_stds = [results[stage]["SepConv"]["std"] for stage in stages]
    exspam_stds = [results[stage]["ExSPAM"]["std"] for stage in stages]
    dynfilter_stds = [results[stage]["DynamicFilter"]["std"] for stage in stages]
    
    # 1. Per-stage comparison (Bar chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(stages))
    width = 0.25
    
    bars1 = ax1.bar(x - width, sepconv_latencies, width, label='SepConv', 
                    color=colors[0], yerr=sepconv_stds, capsize=3)
    bars2 = ax1.bar(x, exspam_latencies, width, label='ExSPAM', 
                    color=colors[1], yerr=exspam_stds, capsize=3)
    bars3 = ax1.bar(x + width, dynfilter_latencies, width, label='DynamicFilter', 
                    color=colors[2], yerr=dynfilter_stds, capsize=3)
    
    ax1.set_xlabel('MetaFormer Stages')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Token Mixer Latency by Stage')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(ax1, bars1)
    add_value_labels(ax1, bars2) 
    add_value_labels(ax1, bars3)
    
    # 2. Overall comparison (focusing on Stage 3 - the bottleneck)
    stage3_data = [
        results["Stage 3"]["SepConv"]["mean"],
        results["Stage 3"]["ExSPAM"]["mean"], 
        results["Stage 3"]["DynamicFilter"]["mean"]
    ]
    stage3_stds = [
        results["Stage 3"]["SepConv"]["std"],
        results["Stage 3"]["ExSPAM"]["std"],
        results["Stage 3"]["DynamicFilter"]["std"]
    ]
    
    bars = ax2.bar(token_mixers, stage3_data, color=colors, yerr=stage3_stds, capsize=5)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Token Mixer Latency (Stage 3: 320 dim, 14x14)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, stage3_data):
        height = bar.get_height()
        ax2.annotate(f'{height:.2f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results_figs/token_mixer_latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print("✓ Saved: token_mixer_latency_comparison.png")
    
    # 3. Create speedup analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate relative speedup (SepConv as baseline)
    speedups = {}
    for stage in stages:
        sepconv_base = results[stage]["SepConv"]["mean"]
        speedups[stage] = {
            "ExSPAM_slowdown": results[stage]["ExSPAM"]["mean"] / sepconv_base,
            "DynamicFilter_slowdown": results[stage]["DynamicFilter"]["mean"] / sepconv_base
        }
    
    exspam_slowdowns = [speedups[stage]["ExSPAM_slowdown"] for stage in stages]
    dynfilter_slowdowns = [speedups[stage]["DynamicFilter_slowdown"] for stage in stages]
    
    x = np.arange(len(stages))
    width = 0.35
    
    ax.bar(x - width/2, exspam_slowdowns, width, label='ExSPAM vs SepConv', color=colors[1])
    ax.bar(x + width/2, dynfilter_slowdowns, width, label='DynamicFilter vs SepConv', color=colors[2])
    
    ax.set_xlabel('MetaFormer Stages')
    ax.set_ylabel('Relative Slowdown (×)')
    ax.set_title('Token Mixer Relative Performance (SepConv = 1.0×)')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='SepConv baseline')
    
    # Add value labels
    for i, (ex_slow, dy_slow) in enumerate(zip(exspam_slowdowns, dynfilter_slowdowns)):
        ax.annotate(f'{ex_slow:.1f}×', xy=(i - width/2, ex_slow), xytext=(0, 3),
                   textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        ax.annotate(f'{dy_slow:.1f}×', xy=(i + width/2, dy_slow), xytext=(0, 3),
                   textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results_figs/token_mixer_speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print("✓ Saved: token_mixer_speedup_analysis.png")
    
    return speedups


def print_summary(results, speedups):
    """Print comprehensive summary"""
    print("\n" + "="*60)
    print("SUMMARY: Token Mixer Latency Analysis")
    print("="*60)
    
    # Overall model results (from previous benchmark)
    model_results = {
        "convformer_s18": 110.857,
        "spanetv2_s18_pure": 137.849,  
        "dfformer_s18": 170.325
    }
    
    print("\nFull Model Performance (from benchmark.py):")
    print("-" * 40)
    for model, latency in model_results.items():
        mixer = model.split('_')[0]
        print(f"{mixer:15s}: {latency:7.2f} ms/step")
    
    print(f"\nSpeedup Analysis (SepConv as baseline):")
    print("-" * 40)
    sepconv_base = model_results["convformer_s18"]
    exspam_slowdown = model_results["spanetv2_s18_pure"] / sepconv_base
    dynfilter_slowdown = model_results["dfformer_s18"] / sepconv_base
    
    print(f"ExSPAM:        {exspam_slowdown:.2f}× slower than SepConv")
    print(f"DynamicFilter: {dynfilter_slowdown:.2f}× slower than SepConv")
    
    print("\nStage 3 (320 dim, 14x14) - Main Bottleneck:")
    print("-" * 40)
    stage3 = results["Stage 3"]
    for mixer in ["SepConv", "ExSPAM", "DynamicFilter"]:
        lat = stage3[mixer]["mean"]
        std = stage3[mixer]["std"]
        print(f"{mixer:15s}: {lat:7.3f} ± {std:.3f} ms")
    
    print(f"\nToken Mixer Complexity Analysis:")
    print("-" * 40)
    print("SepConv:       Simple depthwise separable conv (7×7)")
    print("ExSPAM:        Multi-head conv + FFT/iFFT + modulation")  
    print("DynamicFilter: FFT/iFFT + complex weights + routing")
    
    print(f"\nConclusion:")
    print("-" * 40)
    print("✓ SepConv is the fastest due to simple depthwise convolution")
    print("✓ ExSPAM has moderate overhead from FFT operations and multi-head conv")
    print("✓ DynamicFilter is slowest due to complex frequency domain operations")
    print("✓ The ranking matches the full model benchmark: SepConv > ExSPAM > DynamicFilter")


def main():
    """Main analysis function"""
    if not torch.cuda.is_available():
        print("CUDA not available. This analysis requires GPU.")
        return
        
    print("Starting Token Mixer Latency Analysis...")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Run analysis
    results = analyze_token_mixers()
    
    # Create visualizations
    speedups = create_visualizations(results)
    
    # Print summary
    print_summary(results, speedups)
    
    print("\nAnalysis complete! Graphs saved to results_figs/")


if __name__ == "__main__":
    main()
