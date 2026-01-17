"""
COCO-style Evaluation for Object Size Metrics.

This module provides COCO-style evaluation to get mAP breakdown
by object size (small, medium, large).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


@dataclass
class SizeMetrics:
    """Metrics broken down by object size."""

    model_name: str
    ap_small: float      # Objects < 32x32 pixels
    ap_medium: float     # Objects 32x32 to 96x96 pixels
    ap_large: float      # Objects > 96x96 pixels
    ap_overall: float    # Overall AP @ IoU=0.50:0.95

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "AP_small": round(self.ap_small, 4),
            "AP_medium": round(self.ap_medium, 4),
            "AP_large": round(self.ap_large, 4),
            "AP_overall": round(self.ap_overall, 4),
        }


class COCOEvaluator:
    """COCO-style evaluator for size-based metrics."""

    # VisDrone class mapping to COCO format
    VISDRONE_CLASSES = [
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor"
    ]

    def __init__(self, data_path: str | Path, imgsz: int = 640):
        """
        Initialize COCO evaluator.

        Args:
            data_path: Path to VisDrone dataset
            imgsz: Image size for inference
        """
        self.data_path = Path(data_path)
        self.imgsz = imgsz

    def create_coco_annotations(self, split: str = "val") -> dict:
        """
        Create COCO-format annotations from VisDrone labels.

        Args:
            split: Dataset split (train, val, test)

        Returns:
            COCO annotations dictionary
        """
        split_map = {
            "train": "VisDrone2019-DET-train",
            "val": "VisDrone2019-DET-val",
            "test": "VisDrone2019-DET-test-dev",
        }

        split_dir = self.data_path / split_map[split]
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": name}
                for i, name in enumerate(self.VISDRONE_CLASSES)
            ],
        }

        annotation_id = 0
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

        for img_id, img_path in enumerate(image_files):
            # Get image dimensions (assuming standard VisDrone resolution)
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size

            coco_dict["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            })

            # Read YOLO format labels
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center = float(parts[1]) * width
                            y_center = float(parts[2]) * height
                            box_width = float(parts[3]) * width
                            box_height = float(parts[4]) * height

                            x_min = x_center - box_width / 2
                            y_min = y_center - box_height / 2

                            coco_dict["annotations"].append({
                                "id": annotation_id,
                                "image_id": img_id,
                                "category_id": cls_id,
                                "bbox": [x_min, y_min, box_width, box_height],
                                "area": box_width * box_height,
                                "iscrowd": 0,
                            })
                            annotation_id += 1

        return coco_dict

    def generate_predictions(
        self,
        model_path: str | Path,
        split: str = "val",
        device: int | str = 0,
    ) -> list[dict]:
        """
        Generate COCO-format predictions from a YOLO model.

        Args:
            model_path: Path to model weights
            split: Dataset split
            device: Inference device

        Returns:
            List of COCO-format predictions
        """
        model = YOLO(str(model_path))

        split_map = {
            "train": "VisDrone2019-DET-train",
            "val": "VisDrone2019-DET-val",
            "test": "VisDrone2019-DET-test-dev",
        }

        split_dir = self.data_path / split_map[split]
        images_dir = split_dir / "images"

        predictions = []
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

        print(f"Generating predictions for {len(image_files)} images...")

        for img_id, img_path in enumerate(image_files):
            results = model.predict(str(img_path), device=device, verbose=False)[0]

            if results.boxes is not None and len(results.boxes):
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()

                for box, score, cls in zip(boxes, scores, classes):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min

                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [float(x_min), float(y_min), float(width), float(height)],
                        "score": float(score),
                    })

        return predictions

    def evaluate(
        self,
        model_path: str | Path,
        model_name: str | None = None,
        split: str = "val",
        device: int | str = 0,
        cache_dir: str | Path | None = None,
    ) -> SizeMetrics:
        """
        Run COCO evaluation and get size-based metrics.

        Args:
            model_path: Path to model weights
            model_name: Optional model name
            split: Dataset split
            device: Inference device
            cache_dir: Optional cache directory for annotations

        Returns:
            SizeMetrics with AP breakdown by object size
        """
        model_path = Path(model_path)
        if model_name is None:
            model_name = model_path.stem

        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning COCO evaluation for {model_name}...")

        # Create or load ground truth annotations
        gt_cache = cache_dir / f"coco_gt_{split}.json" if cache_dir else None
        if gt_cache and gt_cache.exists():
            print("Loading cached ground truth annotations...")
            coco_gt = COCO(str(gt_cache))
        else:
            print("Creating ground truth annotations...")
            gt_dict = self.create_coco_annotations(split)
            if gt_cache:
                with open(gt_cache, "w") as f:
                    json.dump(gt_dict, f)
            # Create COCO object from dict
            coco_gt = COCO()
            coco_gt.dataset = gt_dict
            coco_gt.createIndex()

        # Generate predictions
        predictions = self.generate_predictions(model_path, split, device)

        # Save predictions if cache_dir provided
        if cache_dir:
            pred_path = cache_dir / f"predictions_{model_name}_{split}.json"
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

        # Load results
        coco_dt = coco_gt.loadRes(predictions)

        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics by size
        # stats indices: 0=AP, 1=AP50, 2=AP75, 3=AP_small, 4=AP_medium, 5=AP_large
        stats = coco_eval.stats

        return SizeMetrics(
            model_name=model_name,
            ap_small=float(stats[3]) if not np.isnan(stats[3]) else 0.0,
            ap_medium=float(stats[4]) if not np.isnan(stats[4]) else 0.0,
            ap_large=float(stats[5]) if not np.isnan(stats[5]) else 0.0,
            ap_overall=float(stats[0]) if not np.isnan(stats[0]) else 0.0,
        )


def evaluate_by_size(
    model_path: str | Path,
    data_path: str | Path,
    model_name: str | None = None,
    split: str = "val",
    device: int | str = 0,
) -> SizeMetrics:
    """
    Evaluate model with COCO metrics broken down by object size.

    Args:
        model_path: Path to model weights
        data_path: Path to VisDrone dataset
        model_name: Optional model name
        split: Dataset split
        device: Inference device

    Returns:
        SizeMetrics instance
    """
    evaluator = COCOEvaluator(data_path)
    return evaluator.evaluate(
        model_path=model_path,
        model_name=model_name,
        split=split,
        device=device,
    )


def main():
    """CLI entry point for COCO evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="COCO-style evaluation for VisDrone")
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
        "--data",
        type=str,
        default="datasets/VisDrone",
        help="Path to VisDrone dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/coco_eval.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    import pandas as pd

    evaluator = COCOEvaluator(args.data)

    results = []

    # Evaluate YOLO26
    yolo26_metrics = evaluator.evaluate(
        args.yolo26_weights,
        model_name="YOLO26n",
        split=args.split,
        device=args.device,
    )
    results.append(yolo26_metrics.to_dict())

    # Evaluate YOLO11
    yolo11_metrics = evaluator.evaluate(
        args.yolo11_weights,
        model_name="YOLO11n",
        split=args.split,
        device=args.device,
    )
    results.append(yolo11_metrics.to_dict())

    # Print comparison
    print("\n" + "=" * 70)
    print("COCO Evaluation Results (by Object Size)")
    print("=" * 70)
    print(f"{'Metric':<15} {'YOLO26n':<15} {'YOLO11n':<15} {'Diff':<15}")
    print("-" * 70)

    for metric in ["AP_small", "AP_medium", "AP_large", "AP_overall"]:
        y26 = results[0][metric]
        y11 = results[1][metric]
        diff = y26 - y11
        sign = "+" if diff > 0 else ""
        print(f"{metric:<15} {y26:<15.4f} {y11:<15.4f} {sign}{diff:.4f}")

    print("=" * 70)

    # Save results
    df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
