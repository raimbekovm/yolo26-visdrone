# YOLO26 vs YOLO11 Benchmark on VisDrone Dataset

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![YOLO26](https://img.shields.io/badge/Model-YOLO26-brightgreen)](https://docs.ultralytics.com/models/yolo26/)
[![HuggingFace Demo](https://img.shields.io/badge/HuggingFace-Demo-yellow)](https://huggingface.co/spaces/raimbekovm/yolo26-visdrone)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/)

**Independent benchmark comparing YOLO26 and YOLO11 on VisDrone dataset for small object detection from drone imagery.**

## Overview

This project independently verifies Ultralytics' claims about YOLO26:

1. **43% faster CPU inference** compared to YOLO11
2. **Improved small object detection** via Progressive Loss (ProgLoss) and Spatial-channel Token Attention Layer (STAL)
3. **NMS-free end-to-end inference** with learnable one-to-one assignment

### Why VisDrone?

VisDrone is the ideal benchmark for testing small object detection:
- **6,471 training / 548 validation images** captured by drones
- **10 object classes**: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- **Abundant small objects** due to aerial perspective
- Native Ultralytics support: `yolo train data=VisDrone.yaml`

## Benchmark Results

### Overall Performance

| Model | mAP50 | mAP50-95 | Parameters | GFLOPs |
|-------|-------|----------|------------|--------|
| YOLO26n | TBD | TBD | TBD | TBD |
| YOLO11n | TBD | TBD | TBD | TBD |

### Performance by Object Size (pycocotools)

| Model | AP (small <32px) | AP (medium 32-96px) | AP (large >96px) |
|-------|------------------|---------------------|------------------|
| YOLO26n | TBD | TBD | TBD |
| YOLO11n | TBD | TBD | TBD |

### Inference Speed

| Model | GPU (T4) ms | CPU ms | Speedup |
|-------|-------------|--------|---------|
| YOLO26n | TBD | TBD | - |
| YOLO11n | TBD | TBD | baseline |

## Key Findings

> **Results will be updated after training completion**

- [ ] CPU inference speedup verification (claimed: 43% faster)
- [ ] Small object detection improvement (ProgLoss + STAL)
- [ ] NMS-free inference latency benefits
- [ ] Memory efficiency comparison

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/raimbekovm/yolo26-visdrone.git
cd yolo26-visdrone

# Install dependencies
pip install -e .
# or
pip install -r requirements.txt
```

### Training

```bash
# Train YOLO26n on VisDrone
make train-yolo26

# Train YOLO11n on VisDrone
make train-yolo11

# Or manually:
yolo detect train model=yolo26n.pt data=VisDrone.yaml epochs=100 imgsz=640 batch=16
yolo detect train model=yolo11n.pt data=VisDrone.yaml epochs=100 imgsz=640 batch=16
```

### Evaluation

```bash
# Run full benchmark
make benchmark

# Speed benchmark only
make speed-test

# COCO evaluation with size metrics
make coco-eval
```

### Run on Kaggle

The easiest way to reproduce results is through our Kaggle notebook:

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/)

```bash
# Or run locally
jupyter notebook notebooks/kaggle_visdrone_benchmark.ipynb
```

## Project Structure

```
yolo26-visdrone/
├── app/
│   └── gradio_app.py          # HuggingFace Spaces demo
├── configs/
│   ├── config.yaml            # Hydra main config
│   ├── data/visdrone.yaml
│   ├── model/                 # YOLO26n, YOLO11n configs
│   └── training/              # Training configurations
├── data/
│   └── visdrone.yaml          # Dataset config
├── docker/
│   └── Dockerfile
├── notebooks/
│   └── kaggle_visdrone_benchmark.ipynb
├── src/
│   ├── data/download.py       # VisDrone download
│   ├── evaluation/
│   │   ├── benchmark.py       # Speed benchmark
│   │   ├── metrics.py         # mAP by object size
│   │   └── coco_eval.py       # COCO-style evaluation
│   ├── training/trainer.py
│   └── utils/
│       ├── constants.py
│       └── visualization.py   # Comparison plots
├── tests/
├── app.py                     # HuggingFace entry point
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Comparison Visualizations

### mAP by Object Size
![mAP Comparison](assets/map_by_size.png)
*Chart will be generated after training*

### Inference Speed
![Speed Comparison](assets/speed_comparison.png)
*Chart will be generated after training*

## YOLO26 Key Features

YOLO26 introduces several architectural improvements:

1. **Area Attention (AA)** - Efficient spatial attention mechanism
2. **ProgLoss** - Progressive loss scaling for small object detection
3. **STAL** - Spatial-channel Token Attention Layer
4. **NMS-free** - End-to-end detection without post-processing

## Dataset Details

### VisDrone Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | pedestrian | Walking person |
| 1 | people | Standing/sitting person |
| 2 | bicycle | Bicycle |
| 3 | car | Car |
| 4 | van | Van |
| 5 | truck | Truck |
| 6 | tricycle | Tricycle |
| 7 | awning-tricycle | Tricycle with awning |
| 8 | bus | Bus |
| 9 | motor | Motorcycle |

### Statistics

- **Training images**: 6,471
- **Validation images**: 548
- **Test images**: 1,610 (dev set)
- **Image resolution**: Various (typically 1920x1080)
- **Annotation format**: YOLO txt (converted automatically)

## References

- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [VisDrone Dataset - Ultralytics](https://docs.ultralytics.com/datasets/detect/visdrone/)
- [VisDrone Challenge](http://aiskyeye.com/)
- [pycocotools for size-based metrics](https://github.com/cocodataset/cocoapi)

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{yolo26-visdrone-benchmark,
  author = {Murat Raimbekov},
  title = {YOLO26 vs YOLO11 Benchmark on VisDrone Dataset},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/raimbekovm/yolo26-visdrone}
}

@article{visdrone2019,
  title={VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results},
  author={Du, Dawei and others},
  booktitle={ICCVW},
  year={2019}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Author

**Murat Raimbekov**
- Email: murat.raimbekov2004@gmail.com
- GitHub: [@raimbekovm](https://github.com/raimbekovm)

---

**Disclaimer**: This is an independent benchmark. Results may vary based on hardware, software versions, and training conditions.
