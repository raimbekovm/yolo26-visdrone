"""Training module for YOLO models on VisDrone."""

from .trainer import YOLOTrainer, train_model

__all__ = ["YOLOTrainer", "train_model"]
