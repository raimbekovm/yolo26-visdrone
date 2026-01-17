"""
Metrics Calculation for YOLO Models.

This module provides comprehensive metrics calculation including
mAP at different IoU thresholds and object sizes.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""

    model_name: str
    map50: float
    map50_95: float
    precision: float
    recall: float
    parameters: int
    gflops: float

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "mAP50": round(self.map50, 4),
            "mAP50-95": round(self.map50_95, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "Parameters (M)": round(self.parameters / 1e6, 2),
            "GFLOPs": round(self.gflops, 2),
        }


class MetricsCalculator:
    """Calculate and compare metrics for YOLO models."""

    def __init__(self, data_yaml: str = "VisDrone.yaml", imgsz: int = 640, device: int | str = 0):
        """
        Initialize metrics calculator.

        Args:
            data_yaml: Path to dataset YAML config
            imgsz: Image size for validation
            device: Device for inference
        """
        self.data_yaml = data_yaml
        self.imgsz = imgsz
        self.device = device

    def evaluate_model(
        self,
        model_path: str | Path,
        model_name: str | None = None,
    ) -> DetectionMetrics:
        """
        Evaluate a single model.

        Args:
            model_path: Path to model weights
            model_name: Optional name for the model

        Returns:
            DetectionMetrics with evaluation results
        """
        model_path = Path(model_path)
        if model_name is None:
            model_name = model_path.stem

        # Load model
        model = YOLO(str(model_path))

        # Run validation
        print(f"\nEvaluating {model_name}...")
        results = model.val(
            data=self.data_yaml,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        # Extract metrics
        metrics = results.box

        # Get model info
        model_info = model.info(verbose=False)
        parameters = model_info[0] if isinstance(model_info, tuple) else 0
        gflops = model_info[1] if isinstance(model_info, tuple) and len(model_info) > 1 else 0

        return DetectionMetrics(
            model_name=model_name,
            map50=float(metrics.map50),
            map50_95=float(metrics.map),
            precision=float(metrics.mp),
            recall=float(metrics.mr),
            parameters=int(parameters) if parameters else 0,
            gflops=float(gflops) if gflops else 0,
        )

    def compare_models(
        self,
        yolo26_path: str | Path,
        yolo11_path: str | Path,
    ) -> pd.DataFrame:
        """
        Compare YOLO26 and YOLO11 models.

        Args:
            yolo26_path: Path to YOLO26 weights
            yolo11_path: Path to YOLO11 weights

        Returns:
            DataFrame with comparison results
        """
        results = []

        # Evaluate YOLO26
        yolo26_metrics = self.evaluate_model(yolo26_path, "YOLO26n")
        results.append(yolo26_metrics.to_dict())

        # Evaluate YOLO11
        yolo11_metrics = self.evaluate_model(yolo11_path, "YOLO11n")
        results.append(yolo11_metrics.to_dict())

        # Print comparison
        print("\n" + "=" * 60)
        print("Model Comparison Results")
        print("=" * 60)
        print(f"{'Metric':<15} {'YOLO26n':<15} {'YOLO11n':<15} {'Diff':<15}")
        print("-" * 60)

        for metric in ["mAP50", "mAP50-95", "Precision", "Recall"]:
            y26 = results[0][metric]
            y11 = results[1][metric]
            diff = y26 - y11
            sign = "+" if diff > 0 else ""
            print(f"{metric:<15} {y26:<15.4f} {y11:<15.4f} {sign}{diff:.4f}")

        print("=" * 60)

        return pd.DataFrame(results)


def calculate_metrics(
    model_path: str | Path,
    data_yaml: str = "VisDrone.yaml",
    imgsz: int = 640,
    device: int | str = 0,
) -> DetectionMetrics:
    """
    Calculate metrics for a single model.

    Args:
        model_path: Path to model weights
        data_yaml: Dataset configuration
        imgsz: Image size
        device: Device for inference

    Returns:
        DetectionMetrics instance
    """
    calculator = MetricsCalculator(
        data_yaml=data_yaml,
        imgsz=imgsz,
        device=device,
    )
    return calculator.evaluate_model(model_path)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate YOLO metrics")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="VisDrone.yaml",
        help="Dataset config (default: VisDrone.yaml)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (default: 0)",
    )
    args = parser.parse_args()

    metrics = calculate_metrics(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("\nMetrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
