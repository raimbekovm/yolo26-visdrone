"""
Speed Benchmark for YOLO Models.

This module provides comprehensive speed benchmarking for YOLO models
on both CPU and GPU devices.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    device: str
    warmup_runs: int
    test_runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    fps: float

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "mean_ms": round(self.mean_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "fps": round(self.fps, 1),
        }


class BenchmarkRunner:
    """Run speed benchmarks for YOLO models."""

    def __init__(
        self,
        warmup_runs: int = 10,
        test_runs: int = 100,
        imgsz: int = 640,
    ):
        """
        Initialize benchmark runner.

        Args:
            warmup_runs: Number of warmup iterations
            test_runs: Number of test iterations
            imgsz: Input image size
        """
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.imgsz = imgsz

    def benchmark_model(
        self,
        model_path: str | Path,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        model_name: str | None = None,
    ) -> BenchmarkResult:
        """
        Benchmark a single model.

        Args:
            model_path: Path to model weights
            device: Device to run inference on
            model_name: Optional name for the model

        Returns:
            BenchmarkResult with timing statistics
        """
        model_path = Path(model_path)
        if model_name is None:
            model_name = model_path.stem

        # Load model
        model = YOLO(str(model_path))

        # Create dummy input
        dummy_input = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)

        # Warmup
        print(f"Warming up {model_name} on {device}...")
        for _ in range(self.warmup_runs):
            model.predict(dummy_input, device=device, verbose=False)

        # Synchronize GPU if using CUDA
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        print(f"Benchmarking {model_name} ({self.test_runs} runs)...")
        times = []
        for _ in range(self.test_runs):
            start = time.perf_counter()
            model.predict(dummy_input, device=device, verbose=False)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)

        return BenchmarkResult(
            model_name=model_name,
            device=device,
            warmup_runs=self.warmup_runs,
            test_runs=self.test_runs,
            mean_ms=float(np.mean(times)),
            std_ms=float(np.std(times)),
            min_ms=float(np.min(times)),
            max_ms=float(np.max(times)),
            fps=1000.0 / float(np.mean(times)),
        )

    def compare_models(
        self,
        yolo26_path: str | Path,
        yolo11_path: str | Path,
        devices: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare YOLO26 and YOLO11 models.

        Args:
            yolo26_path: Path to YOLO26 weights
            yolo11_path: Path to YOLO11 weights
            devices: List of devices to benchmark on

        Returns:
            DataFrame with comparison results
        """
        if devices is None:
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda")
            elif torch.backends.mps.is_available():
                devices.append("mps")

        results = []

        for device in devices:
            print(f"\n{'=' * 50}")
            print(f"Benchmarking on {device.upper()}")
            print("=" * 50)

            # Benchmark YOLO26
            yolo26_result = self.benchmark_model(
                yolo26_path, device=device, model_name="YOLO26n"
            )
            results.append(yolo26_result.to_dict())

            # Benchmark YOLO11
            yolo11_result = self.benchmark_model(
                yolo11_path, device=device, model_name="YOLO11n"
            )
            results.append(yolo11_result.to_dict())

            # Calculate speedup
            speedup = yolo11_result.mean_ms / yolo26_result.mean_ms
            print(f"\nYOLO26 speedup vs YOLO11 on {device}: {speedup:.2f}x")
            print(f"YOLO26: {yolo26_result.mean_ms:.2f}ms ({yolo26_result.fps:.1f} FPS)")
            print(f"YOLO11: {yolo11_result.mean_ms:.2f}ms ({yolo11_result.fps:.1f} FPS)")

        return pd.DataFrame(results)


def run_speed_benchmark(
    yolo26_path: str | Path,
    yolo11_path: str | Path,
    warmup_runs: int = 10,
    test_runs: int = 100,
    imgsz: int = 640,
    devices: list[str] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run speed benchmark comparing YOLO26 and YOLO11.

    Args:
        yolo26_path: Path to YOLO26 model weights
        yolo11_path: Path to YOLO11 model weights
        warmup_runs: Number of warmup iterations
        test_runs: Number of test iterations
        imgsz: Input image size
        devices: List of devices to benchmark on
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with benchmark results
    """
    runner = BenchmarkRunner(
        warmup_runs=warmup_runs,
        test_runs=test_runs,
        imgsz=imgsz,
    )

    results_df = runner.compare_models(
        yolo26_path=yolo26_path,
        yolo11_path=yolo11_path,
        devices=devices,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    return results_df


def main():
    """CLI entry point for benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Speed Benchmark")
    parser.add_argument(
        "--yolo26-weights",
        type=str,
        required=True,
        help="Path to YOLO26 weights",
    )
    parser.add_argument(
        "--yolo11-weights",
        type=str,
        required=True,
        help="Path to YOLO11 weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Warmup runs (default: 10)",
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=100,
        help="Test runs (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device(s) to benchmark, comma-separated (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/speed_benchmark.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--speed-only",
        action="store_true",
        help="Run speed benchmark only (no accuracy evaluation)",
    )
    args = parser.parse_args()

    devices = args.device.split(",") if args.device else None

    run_speed_benchmark(
        yolo26_path=args.yolo26_weights,
        yolo11_path=args.yolo11_weights,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs,
        imgsz=args.imgsz,
        devices=devices,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
