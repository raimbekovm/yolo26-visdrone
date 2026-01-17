"""Tests for benchmark module."""

import numpy as np
import pytest


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_import(self):
        """Test that benchmark module can be imported."""
        from src.evaluation.benchmark import BenchmarkRunner, run_speed_benchmark

        assert BenchmarkRunner is not None
        assert run_speed_benchmark is not None

    def test_benchmark_result_to_dict(self):
        """Test BenchmarkResult conversion to dict."""
        from src.evaluation.benchmark import BenchmarkResult

        result = BenchmarkResult(
            model_name="test_model",
            device="cpu",
            warmup_runs=10,
            test_runs=100,
            mean_ms=15.5,
            std_ms=1.2,
            min_ms=12.0,
            max_ms=20.0,
            fps=64.5,
        )

        result_dict = result.to_dict()

        assert result_dict["model"] == "test_model"
        assert result_dict["device"] == "cpu"
        assert result_dict["mean_ms"] == 15.5
        assert result_dict["fps"] == 64.5


class TestMetrics:
    """Tests for metrics module."""

    def test_import(self):
        """Test that metrics module can be imported."""
        from src.evaluation.metrics import MetricsCalculator, calculate_metrics

        assert MetricsCalculator is not None
        assert calculate_metrics is not None

    def test_detection_metrics_to_dict(self):
        """Test DetectionMetrics conversion to dict."""
        from src.evaluation.metrics import DetectionMetrics

        metrics = DetectionMetrics(
            model_name="test_model",
            map50=0.45,
            map50_95=0.32,
            precision=0.55,
            recall=0.48,
            parameters=2600000,
            gflops=6.5,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["model"] == "test_model"
        assert metrics_dict["mAP50"] == 0.45
        assert metrics_dict["mAP50-95"] == 0.32
        assert metrics_dict["Parameters (M)"] == 2.6


class TestCOCOEval:
    """Tests for COCO evaluation module."""

    def test_import(self):
        """Test that coco_eval module can be imported."""
        from src.evaluation.coco_eval import COCOEvaluator, evaluate_by_size

        assert COCOEvaluator is not None
        assert evaluate_by_size is not None

    def test_size_metrics_to_dict(self):
        """Test SizeMetrics conversion to dict."""
        from src.evaluation.coco_eval import SizeMetrics

        metrics = SizeMetrics(
            model_name="test_model",
            ap_small=0.15,
            ap_medium=0.35,
            ap_large=0.55,
            ap_overall=0.32,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["model"] == "test_model"
        assert metrics_dict["AP_small"] == 0.15
        assert metrics_dict["AP_medium"] == 0.35
        assert metrics_dict["AP_large"] == 0.55


class TestConstants:
    """Tests for constants module."""

    def test_visdrone_classes(self):
        """Test VisDrone class definitions."""
        from src.utils.constants import VISDRONE_CLASSES

        assert len(VISDRONE_CLASSES) == 10
        assert "pedestrian" in VISDRONE_CLASSES
        assert "car" in VISDRONE_CLASSES
        assert "bus" in VISDRONE_CLASSES

    def test_size_thresholds(self):
        """Test object size thresholds."""
        from src.utils.constants import SIZE_THRESHOLDS

        assert SIZE_THRESHOLDS["small"] == 32 ** 2
        assert SIZE_THRESHOLDS["medium"] == 96 ** 2


class TestVisualization:
    """Tests for visualization module."""

    def test_import(self):
        """Test that visualization module can be imported."""
        from src.utils.visualization import (
            plot_comparison,
            plot_size_metrics,
            create_benchmark_report,
        )

        assert plot_comparison is not None
        assert plot_size_metrics is not None
        assert create_benchmark_report is not None


class TestTrainer:
    """Tests for trainer module."""

    def test_import(self):
        """Test that trainer module can be imported."""
        from src.training.trainer import YOLOTrainer, TrainingConfig, train_model

        assert YOLOTrainer is not None
        assert TrainingConfig is not None
        assert train_model is not None

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig()

        assert config.epochs == 100
        assert config.batch == 16
        assert config.imgsz == 640
        assert config.data == "VisDrone.yaml"

    def test_training_config_to_dict(self):
        """Test TrainingConfig conversion to dict."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig(epochs=50, batch=32)
        config_dict = config.to_dict()

        assert config_dict["epochs"] == 50
        assert config_dict["batch"] == 32


class TestDataModule:
    """Tests for data module."""

    def test_import(self):
        """Test that data module can be imported."""
        from src.data.download import download_visdrone, verify_dataset

        assert download_visdrone is not None
        assert verify_dataset is not None


# Integration tests (require model files)
@pytest.mark.skipif(True, reason="Requires model files")
class TestIntegration:
    """Integration tests that require actual model files."""

    def test_model_inference(self):
        """Test model inference on dummy image."""
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        results = model.predict(dummy_image, verbose=False)

        assert results is not None
        assert len(results) > 0
