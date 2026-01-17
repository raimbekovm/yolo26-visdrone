"""
YOLO Trainer for VisDrone Dataset.

This module provides a unified interface for training YOLO models
on the VisDrone dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    model: str = "yolo11n.pt"  # Pretrained model

    # Data
    data: str = "VisDrone.yaml"
    imgsz: int = 640

    # Training
    epochs: int = 100
    batch: int = 16
    device: int | str = 0

    # Optimizer
    optimizer: str = "AdamW"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Augmentation
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.0

    # Output
    project: str = "runs/detect"
    name: str | None = None
    exist_ok: bool = True
    plots: bool = True
    save: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
            "model": self.model,
            "data": self.data,
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": self.batch,
            "device": self.device,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "augment": self.augment,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
            "plots": self.plots,
            "save": self.save,
        }


class YOLOTrainer:
    """Unified YOLO trainer for VisDrone benchmark."""

    def __init__(self, config: TrainingConfig | None = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()

    def train(
        self,
        model: str | None = None,
        name: str | None = None,
        **kwargs,
    ) -> tuple[YOLO, Any]:
        """
        Train a YOLO model.

        Args:
            model: Model to train (overrides config)
            name: Run name (overrides config)
            **kwargs: Additional training arguments

        Returns:
            Tuple of (trained model, training results)
        """
        # Merge config with overrides
        train_args = self.config.to_dict()
        if model:
            train_args["model"] = model
        if name:
            train_args["name"] = name
        train_args.update(kwargs)

        # Initialize model
        model_obj = YOLO(train_args.pop("model"))

        # Train
        print(f"\nTraining {model_obj.model_name or 'model'}...")
        print(f"Config: epochs={train_args['epochs']}, batch={train_args['batch']}, "
              f"imgsz={train_args['imgsz']}")

        results = model_obj.train(**train_args)

        return model_obj, results

    def train_yolo26(self, epochs: int | None = None, **kwargs) -> tuple[YOLO, Any]:
        """Train YOLO26n on VisDrone."""
        return self.train(
            model="yolo26n.pt",
            name="yolo26n_visdrone",
            epochs=epochs or self.config.epochs,
            **kwargs,
        )

    def train_yolo11(self, epochs: int | None = None, **kwargs) -> tuple[YOLO, Any]:
        """Train YOLO11n on VisDrone."""
        return self.train(
            model="yolo11n.pt",
            name="yolo11n_visdrone",
            epochs=epochs or self.config.epochs,
            **kwargs,
        )

    def train_both(self, epochs: int | None = None, **kwargs) -> dict:
        """
        Train both YOLO26 and YOLO11 sequentially.

        Args:
            epochs: Number of epochs
            **kwargs: Additional training arguments

        Returns:
            Dictionary with training results for both models
        """
        results = {}

        print("\n" + "=" * 60)
        print("Training YOLO26n on VisDrone")
        print("=" * 60)
        yolo26_model, yolo26_results = self.train_yolo26(epochs=epochs, **kwargs)
        results["yolo26"] = {
            "model": yolo26_model,
            "results": yolo26_results,
            "weights": Path(self.config.project) / "yolo26n_visdrone" / "weights" / "best.pt",
        }

        print("\n" + "=" * 60)
        print("Training YOLO11n on VisDrone")
        print("=" * 60)
        yolo11_model, yolo11_results = self.train_yolo11(epochs=epochs, **kwargs)
        results["yolo11"] = {
            "model": yolo11_model,
            "results": yolo11_results,
            "weights": Path(self.config.project) / "yolo11n_visdrone" / "weights" / "best.pt",
        }

        return results


def train_model(
    model: str = "yolo11n.pt",
    data: str = "VisDrone.yaml",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: int | str = 0,
    name: str | None = None,
    project: str = "runs/detect",
    **kwargs,
) -> tuple[YOLO, Any]:
    """
    Train a YOLO model on VisDrone.

    Args:
        model: Pretrained model to start from
        data: Dataset configuration
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        device: Training device
        name: Run name
        project: Project directory
        **kwargs: Additional training arguments

    Returns:
        Tuple of (trained model, training results)
    """
    config = TrainingConfig(
        model=model,
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        name=name,
        project=project,
    )

    trainer = YOLOTrainer(config)
    return trainer.train(**kwargs)


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO on VisDrone")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        choices=["yolo26n.pt", "yolo11n.pt", "yolo26s.pt", "yolo11s.pt"],
        help="Model to train (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="VisDrone.yaml",
        help="Dataset config (default: VisDrone.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
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
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory",
    )
    parser.add_argument(
        "--train-both",
        action="store_true",
        help="Train both YOLO26 and YOLO11",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        project=args.project,
    )

    trainer = YOLOTrainer(config)

    if args.train_both:
        trainer.train_both()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
