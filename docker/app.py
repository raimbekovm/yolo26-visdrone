"""YOLO26-VisDrone Demo (Standalone for HuggingFace Spaces)"""
import gradio as gr
from ultralytics import YOLO

print("Loading YOLO26n...")
model = YOLO("yolo26n.pt")
print(f"Model loaded! Classes: {len(model.names)}")

def detect(image, conf=0.25, iou=0.45):
    if image is None:
        return None, "Upload an image"
    results = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
    annotated = results.plot()
    annotated = annotated[:, :, ::-1]  # BGR to RGB
    if results.boxes is not None and len(results.boxes) > 0:
        counts = {}
        for c in results.boxes.cls.cpu().numpy():
            name = model.names[int(c)]  # Use model's own class names
            counts[name] = counts.get(name, 0) + 1
        text = f"**{len(results.boxes)} objects**\n"
        for n, c in sorted(counts.items(), key=lambda x: -x[1]):
            text += f"- {n}: {c}\n"
    else:
        text = "No objects detected"
    return annotated, text

with gr.Blocks(title="YOLO26 VisDrone") as demo:
    gr.Markdown("# YOLO26 Detection Demo\nPretrained on COCO, ready for VisDrone fine-tuning")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Input", type="pil")
            conf = gr.Slider(0.01, 1, 0.25, label="Confidence")
            iou = gr.Slider(0.01, 1, 0.45, label="IoU")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Output")
            txt = gr.Markdown()
    btn.click(detect, [img_in, conf, iou], [img_out, txt])
    gr.Markdown("[GitHub](https://github.com/raimbekovm/yolo26-visdrone) | Author: Murat Raimbekov")

demo.launch(server_name="0.0.0.0", server_port=7860)
