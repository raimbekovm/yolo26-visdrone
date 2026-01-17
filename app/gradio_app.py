"""
Gradio Application for YOLO26-VisDrone Demo.

This module provides a Gradio interface for:
1. Object detection using YOLO26 trained on VisDrone
2. Displaying benchmark results
3. Links to GitHub and documentation

Author: Murat Raimbekov
"""

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

# VisDrone class names
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# Benchmark results (to be updated after training)
BENCHMARK_RESULTS = """
## Benchmark Results (VisDrone Dataset)

### Overall Performance
| Model | mAP50 | mAP50-95 | Parameters |
|-------|-------|----------|------------|
| YOLO26n | TBD | TBD | ~2.6M |
| YOLO11n | TBD | TBD | ~2.6M |

### Performance by Object Size
| Model | AP (small) | AP (medium) | AP (large) |
|-------|------------|-------------|------------|
| YOLO26n | TBD | TBD | TBD |
| YOLO11n | TBD | TBD | TBD |

### Inference Speed
| Model | GPU (T4) | CPU |
|-------|----------|-----|
| YOLO26n | TBD | TBD |
| YOLO11n | TBD | TBD |

*Results will be updated after training completion.*
"""

# Model cache
_model_cache: dict[str, Any] = {}


def load_model(model_name: str = "yolo26n"):
    """Load YOLO model with caching."""
    if model_name not in _model_cache:
        try:
            from ultralytics import YOLO

            # Try to load trained model, fall back to pretrained
            model_paths = [
                f"yolo26n_visdrone.pt",  # Trained model
                f"runs/detect/yolo26n_visdrone/weights/best.pt",
                f"yolo26n.pt",  # Pretrained
            ]

            model = None
            for path in model_paths:
                if Path(path).exists():
                    model = YOLO(path)
                    print(f"Loaded model from: {path}")
                    break

            if model is None:
                # Load pretrained as fallback
                model = YOLO("yolo26n.pt")
                print("Loaded pretrained YOLO26n model")

            _model_cache[model_name] = model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    return _model_cache[model_name]


def detect_objects(
    image: Image.Image | np.ndarray | None,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
) -> tuple[np.ndarray | None, str]:
    """
    Detect objects in the input image.

    Args:
        image: Input image
        confidence: Confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        Tuple of (annotated image, detection summary)
    """
    if image is None:
        return None, "Please upload an image."

    # Load model
    model = load_model()
    if model is None:
        return None, "Error: Could not load model."

    try:
        # Run inference
        results = model.predict(
            image,
            conf=confidence,
            iou=iou_threshold,
            verbose=False,
        )[0]

        # Get annotated image (convert BGR to RGB)
        annotated_image = results.plot()
        annotated_image = annotated_image[:, :, ::-1]

        # Generate summary
        if results.boxes is not None and len(results.boxes) > 0:
            # Count detections per class (use model's class names)
            class_counts: dict[str, int] = {}
            for cls_id in results.boxes.cls.cpu().numpy():
                cls_name = model.names[int(cls_id)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            summary_lines = [f"**Total detections: {len(results.boxes)}**\n"]
            summary_lines.append("| Class | Count |")
            summary_lines.append("|-------|-------|")
            for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                summary_lines.append(f"| {cls_name} | {count} |")

            summary = "\n".join(summary_lines)
        else:
            summary = "No objects detected."

        return annotated_image, summary

    except Exception as e:
        return None, f"Error during detection: {str(e)}"


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""
    with gr.Blocks(
        title="YOLO26-VisDrone Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # YOLO26 Object Detection - VisDrone Dataset

            Detect objects in drone imagery using **YOLO26** trained on the VisDrone dataset.

            YOLO26 introduces improvements for small object detection with:
            - **Area Attention (AA)** - Efficient spatial attention
            - **ProgLoss** - Progressive loss for small objects
            - **STAL** - Spatial-channel Token Attention Layer
            - **NMS-free** - End-to-end detection

            ## VisDrone Classes
            `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )

                with gr.Row():
                    confidence = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.25,
                        step=0.01,
                        label="Confidence Threshold",
                    )
                    iou_threshold = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.45,
                        step=0.01,
                        label="IoU Threshold",
                    )

                detect_btn = gr.Button("Detect Objects", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Detection Results",
                    type="numpy",
                )
                detection_summary = gr.Markdown(
                    label="Detection Summary",
                    value="Upload an image to start detection.",
                )

        # Connect detection function
        detect_btn.click(
            fn=detect_objects,
            inputs=[input_image, confidence, iou_threshold],
            outputs=[output_image, detection_summary],
        )

        # Also trigger on image upload
        input_image.change(
            fn=detect_objects,
            inputs=[input_image, confidence, iou_threshold],
            outputs=[output_image, detection_summary],
        )

        # Benchmark results section
        with gr.Accordion("Benchmark Results", open=False):
            gr.Markdown(BENCHMARK_RESULTS)

        # Links section
        gr.Markdown(
            """
            ---
            ### Links
            - [GitHub Repository](https://github.com/raimbekovm/yolo26-visdrone)
            - [Kaggle Notebook](https://www.kaggle.com/)
            - [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
            - [VisDrone Dataset](https://docs.ultralytics.com/datasets/detect/visdrone/)

            ### Author
            **Murat Raimbekov** | [GitHub](https://github.com/raimbekovm) | [Email](mailto:murat.raimbekov2004@gmail.com)

            ---
            *This demo is part of an independent benchmark comparing YOLO26 and YOLO11 on the VisDrone dataset.*
            """
        )

        # Example images (if available)
        example_dir = Path("examples")
        if example_dir.exists():
            example_images = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
            if example_images:
                gr.Examples(
                    examples=[[str(img)] for img in example_images[:5]],
                    inputs=[input_image],
                    outputs=[output_image, detection_summary],
                    fn=detect_objects,
                    cache_examples=True,
                )

    return demo


# For direct running
if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
