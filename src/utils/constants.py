"""
Constants for YOLO26-VisDrone benchmark.
"""

# VisDrone class names
VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

# Colors for visualization (BGR format for OpenCV compatibility)
VISDRONE_COLORS = [
    (255, 0, 0),      # pedestrian - blue
    (0, 255, 0),      # people - green
    (0, 0, 255),      # bicycle - red
    (255, 255, 0),    # car - cyan
    (255, 0, 255),    # van - magenta
    (0, 255, 255),    # truck - yellow
    (128, 0, 0),      # tricycle - dark blue
    (0, 128, 0),      # awning-tricycle - dark green
    (0, 0, 128),      # bus - dark red
    (128, 128, 0),    # motor - olive
]

# Object size thresholds (COCO standard)
SIZE_THRESHOLDS = {
    "small": 32 ** 2,     # area < 32x32 = 1024 pixels
    "medium": 96 ** 2,    # area < 96x96 = 9216 pixels
    "large": float("inf"),  # area >= 96x96
}

# Model configurations
MODELS = {
    "yolo26n": {
        "weights": "yolo26n.pt",
        "name": "YOLO26n",
        "color": "#2ecc71",  # green
    },
    "yolo11n": {
        "weights": "yolo11n.pt",
        "name": "YOLO11n",
        "color": "#3498db",  # blue
    },
}

# Benchmark settings
DEFAULT_BENCHMARK_CONFIG = {
    "warmup_runs": 10,
    "test_runs": 100,
    "imgsz": 640,
    "batch": 1,
}

# Training defaults
DEFAULT_TRAINING_CONFIG = {
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "optimizer": "AdamW",
    "lr0": 0.01,
}

# Dataset statistics
VISDRONE_STATS = {
    "train_images": 6471,
    "val_images": 548,
    "test_images": 1610,
    "total_annotations": 343000,  # approximate
    "classes": 10,
}
