# YOLO26 vs YOLO11 Benchmark on VisDrone Dataset

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/muraraimbekov/yolo26-vs-yolo11-visdrone-benchmark)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Demo-yellow)](https://huggingface.co/spaces/raimbekovm/yolo26-visdrone)

Independent benchmark comparing YOLO26 and YOLO11 on VisDrone dataset for small object detection from drone imagery.

## Claims Under Test

Ultralytics claims YOLO26 delivers:
1. **43% faster CPU inference** compared to YOLO11
2. **Improved small object detection** via ProgLoss and STAL
3. **Smaller model size** due to DFL removal

## Results

Trained on Kaggle T4 GPU | 50 epochs | batch=16 | imgsz=640

### Model Size

| Model | Parameters | ONNX Size |
|-------|------------|-----------|
| YOLO26n | 2.51M | 9.8 MB |
| YOLO11n | 2.59M | 10.6 MB |

### Accuracy

| Model | mAP50 | mAP50-95 | AP_small | AP_medium | AP_large |
|-------|-------|----------|----------|-----------|----------|
| YOLO26n | 0.278 | 0.157 | 0.0425 | 0.182 | 0.302 |
| YOLO11n | 0.302 | 0.172 | 0.0506 | 0.203 | 0.340 |

### Inference Speed

| Model | GPU (T4) | CPU (ONNX) |
|-------|----------|------------|
| YOLO26n | 10.47 ms | 101.14 ms |
| YOLO11n | 9.41 ms | 79.53 ms |

## Verification Summary

| Claim | Result | Verified |
|-------|--------|----------|
| 43% faster CPU inference | YOLO26 is 27% slower | No |
| Better small object detection | YOLO11 AP_small is 19% higher | No |
| Smaller model size | YOLO26 is 3% smaller | Yes |

## Key Findings

1. **CPU Speed**: Contrary to claims, YOLO26n is 27% slower than YOLO11n on CPU (ONNX runtime)
2. **Small Objects**: YOLO11n outperforms YOLO26n on AP_small by 19% (0.0506 vs 0.0425)
3. **Overall Accuracy**: YOLO11n achieves higher mAP50 (0.302 vs 0.278)
4. **GPU Speed**: YOLO11n is 11% faster on T4 GPU
5. **Model Size**: YOLO26n is slightly smaller (2.51M vs 2.59M parameters)

## Methodology

- **Dataset**: VisDrone2019-DET (6,471 train / 548 val images)
- **Hardware**: Kaggle T4 GPU
- **Training**: 50 epochs, batch=16, imgsz=640
- **CPU Benchmark**: ONNX Runtime (as per Ultralytics methodology)
- **Evaluation**: COCO metrics with AP by object size

## Quick Start

```bash
# Clone repository
git clone https://github.com/raimbekovm/yolo26-visdrone.git
cd yolo26-visdrone

# Install dependencies
pip install ultralytics pycocotools onnxruntime

# Train models
yolo detect train model=yolo26n.pt data=VisDrone.yaml epochs=50 imgsz=640 batch=16
yolo detect train model=yolo11n.pt data=VisDrone.yaml epochs=50 imgsz=640 batch=16
```

## Reproduce on Kaggle

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/muraraimbekov/yolo26-vs-yolo11-visdrone-benchmark)

## Dataset

VisDrone is captured by drone-mounted cameras and contains abundant small objects:

| Class | ID |
|-------|-----|
| pedestrian | 0 |
| people | 1 |
| bicycle | 2 |
| car | 3 |
| van | 4 |
| truck | 5 |
| tricycle | 6 |
| awning-tricycle | 7 |
| bus | 8 |
| motor | 9 |

## References

- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [VisDrone Dataset](https://docs.ultralytics.com/datasets/detect/visdrone/)
- [VisDrone Challenge](http://aiskyeye.com/)

## Citation

```bibtex
@misc{yolo26-visdrone-benchmark,
  author = {Murat Raimbekov},
  title = {YOLO26 vs YOLO11 Benchmark on VisDrone Dataset},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/raimbekovm/yolo26-visdrone}
}
```

## License

Apache License 2.0

## Author

Murat Raimbekov
- GitHub: [@raimbekovm](https://github.com/raimbekovm)
- Email: murat.raimbekov2004@gmail.com
