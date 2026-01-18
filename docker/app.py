"""YOLO26 vs YOLO11 Comparison Demo - Fine-tuned on VisDrone"""
import gradio as gr
from ultralytics import YOLO

print("Loading models fine-tuned on VisDrone...")
yolo26 = YOLO("yolo26n_visdrone.pt")
yolo11 = YOLO("yolo11n_visdrone.pt")
print("Models loaded!")

CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']


def format_results(results, model_name):
    if results.boxes is not None and len(results.boxes) > 0:
        counts = {}
        for c in results.boxes.cls.cpu().numpy():
            name = CLASSES[int(c)] if int(c) < len(CLASSES) else f"class_{int(c)}"
            counts[name] = counts.get(name, 0) + 1
        text = f"**{model_name}: {len(results.boxes)} objects**\n"
        for n, c in sorted(counts.items(), key=lambda x: -x[1]):
            text += f"- {n}: {c}\n"
    else:
        text = f"**{model_name}:** No objects detected"
    return text


def detect(image, conf=0.25, iou=0.45):
    if image is None:
        return None, None, "Upload an image"

    res26 = yolo26.predict(image, conf=conf, iou=iou, verbose=False)[0]
    res11 = yolo11.predict(image, conf=conf, iou=iou, verbose=False)[0]

    img26 = res26.plot()[:, :, ::-1]
    img11 = res11.plot()[:, :, ::-1]

    text = format_results(res26, "YOLO26n") + "\n" + format_results(res11, "YOLO11n")

    n26 = len(res26.boxes) if res26.boxes is not None else 0
    n11 = len(res11.boxes) if res11.boxes is not None else 0
    if n26 != n11:
        diff = n26 - n11
        text += f"\n\n*Difference: YOLO26 detected {diff:+d} objects*"

    return img26, img11, text


with gr.Blocks(title="YOLO26 vs YOLO11 VisDrone") as demo:
    gr.Markdown("""# YOLO26 vs YOLO11 on VisDrone

Compare object detection between YOLO26n and YOLO11n, both fine-tuned on VisDrone dataset (drone imagery).

**Benchmark results:** YOLO11n achieved higher mAP (0.302 vs 0.278) and better small object detection.
""")

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(label="Input Image", type="pil")
            conf = gr.Slider(0.01, 1, 0.25, label="Confidence Threshold")
            iou = gr.Slider(0.01, 1, 0.45, label="IoU Threshold")
            btn = gr.Button("Compare", variant="primary")

    with gr.Row():
        img26 = gr.Image(label="YOLO26n (mAP50: 0.278)")
        img11 = gr.Image(label="YOLO11n (mAP50: 0.302)")

    results_text = gr.Markdown()

    btn.click(detect, [img_in, conf, iou], [img26, img11, results_text])

    gr.Markdown("""
---
[GitHub](https://github.com/raimbekovm/yolo26-visdrone) |
[Kaggle Notebook](https://www.kaggle.com/code/muraraimbekov/yolo26-vs-yolo11-visdrone-benchmark) |
Author: Murat Raimbekov
""")

demo.launch(server_name="0.0.0.0", server_port=7860)
