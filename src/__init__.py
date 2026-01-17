"""
YOLO26-VisDrone: Benchmark comparing YOLO26 and YOLO11 on VisDrone dataset.

This package provides tools for:
- Training YOLO models on VisDrone dataset
- Evaluating detection performance with COCO metrics
- Benchmarking inference speed (CPU/GPU)
- Visualizing comparison results
"""

__version__ = "0.1.0"
__author__ = "Murat Raimbekov"
__email__ = "murat.raimbekov2004@gmail.com"

from . import data, evaluation, training, utils

__all__ = ["data", "evaluation", "training", "utils"]
