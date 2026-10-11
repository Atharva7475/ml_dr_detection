"""
Hardware Diagnostic & Workstation Profiling Tool.

Inspects:
  - System OS, Python runtime, CPU cores, RAM.
  - PyTorch installation, CUDA build flags, cuDNN availability.
  - CUDA GPU devices: name, compute capability, total & free VRAM.
  - Mixed precision (AMP) compatibility.
  - Computes tailored batch size recommendations for 224x224 and 384x384 runs.
  - Generates recommended execution commands for tomorrow's workstation.

Usage:
  python scripts/inspect_hardware.py
"""

import sys
import os
import platform
import psutil
import torch


def inspect_hardware():
    print("=" * 75)
    print("         APTOS DR PIPELINE: WORKSTATION HARDWARE DIAGNOSTIC         ")
    print("=" * 75)

    # 1. Operating System & Python Environment
    os_name = platform.system()
    os_release = platform.release()
    os_arch = platform.machine()
    python_ver = sys.version.split()[0]

    print(f"\n[1] HOST ENVIRONMENT:")
    print(f"  Operating System:    {os_name} {os_release} ({os_arch})")
    print(f"  Python Version:      {python_ver}")
    print(f"  CPU Cores (Logical): {psutil.cpu_count(logical=True)}")
    print(f"  CPU Cores (Physical):{psutil.cpu_count(logical=False)}")
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"  Total System RAM:    {ram_gb:.1f} GB (Available: {ram_avail_gb:.1f} GB)")

    # 2. PyTorch & Acceleration Backend
    print(f"\n[2] PYTORCH & ACCELERATION BACKEND:")
    print(f"  PyTorch Version:     {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available:      {cuda_available}")

    gpu_count = 0
    total_vram_gb = 0.0
    vram_class = "cpu"

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"  CUDA Device Count:   {gpu_count}")
        print(f"  CUDA Built Version:  {torch.version.cuda}")
        print(f"  cuDNN Available:     {torch.backends.cudnn.is_available()}")
        if torch.backends.cudnn.is_available():
            print(f"  cuDNN Version:       {torch.backends.cudnn.version()}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            major, minor = props.major, props.minor
            print(f"\n  GPU [{i}]: {props.name}")
            print(f"    Compute Capability: sm_{major}{minor}")
            print(f"    Total VRAM:         {vram_gb:.2f} GB")
            total_vram_gb = max(total_vram_gb, vram_gb)
    else:
        print("  NOTICE: No CUDA GPU detected on this host. Training will run on CPU.")

    # 3. Categorize VRAM & Recommend Batch Sizes
    print(f"\n[3] BATCH SIZE & WORKER RECOMMENDATIONS:")
    if not cuda_available:
        rec_224_bs = 8
        rec_384_bs = 4
        rec_workers = 0 if os_name == "Windows" else 2
        vram_profile = "CPU Only (Debug / Smoke Test mode only)"
    elif total_vram_gb < 6.0:
        rec_224_bs = 16
        rec_384_bs = 8
        rec_workers = 2 if os_name == "Windows" else 4
        vram_profile = f"Low VRAM (<6 GB: {total_vram_gb:.1f} GB)"
    elif total_vram_gb < 10.0:
        rec_224_bs = 32
        rec_384_bs = 16
        rec_workers = 2 if os_name == "Windows" else 4
        vram_profile = f"Standard Mid-Tier VRAM (6-10 GB: {total_vram_gb:.1f} GB, e.g. RTX 3060/4060)"
    elif total_vram_gb < 18.0:
        rec_224_bs = 64
        rec_384_bs = 32
        rec_workers = 4 if os_name == "Windows" else 8
        vram_profile = f"High VRAM (10-18 GB: {total_vram_gb:.1f} GB, e.g. RTX 3080/4070/4080/T4)"
    else:
        rec_224_bs = 128
        rec_384_bs = 48
        rec_workers = 4 if os_name == "Windows" else 8
        vram_profile = f"Ultra / Datacenter VRAM (>18 GB: {total_vram_gb:.1f} GB, e.g. RTX 3090/4090/A100)"

    print(f"  Detected Profile:    {vram_profile}")
    print(f"  Recommended 224x224 Batch Size: {rec_224_bs}")
    print(f"  Recommended 384x384 Batch Size: {rec_384_bs}")
    print(f"  Recommended num_workers:        {rec_workers}")
    print(f"  Automatic Mixed Precision (AMP):{'ENABLED (Supported)' if cuda_available else 'DISABLED (CPU)'}")

    # 4. Tailored Commands for Tomorrow
    print(f"\n[4] EXACT EXECUTION COMMANDS FOR TOMORROW'S WORKSTATION:")
    print("  # Step A: Smoke test before training (verifies GPU setup in <30s):")
    print(f"  python scripts/run_experiments.py --experiment E2 --smoke-test --batch-size {rec_384_bs} --num-workers {rec_workers}")
    print("\n  # Step B: Execute target experiments independently:")
    print(f"  python scripts/run_experiments.py --experiment E1 --batch-size {rec_384_bs} --num-workers {rec_workers}")
    print(f"  python scripts/run_experiments.py --experiment E2 --batch-size {rec_384_bs} --num-workers {rec_workers}")
    print(f"  python scripts/run_experiments.py --experiment E3 --batch-size {rec_384_bs} --num-workers {rec_workers}")
    print(f"  python scripts/run_experiments.py --experiment E4 --batch-size {rec_384_bs} --num-workers {rec_workers}")
    print("\n  # Step C: Compare validation metrics after runs finish:")
    print("  python scripts/compare_experiments.py")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    inspect_hardware()
