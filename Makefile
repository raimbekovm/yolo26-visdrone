.PHONY: help install install-dev download-data train-yolo26 train-yolo11 train-all validate benchmark speed-test coco-eval visualize clean format lint test docker-build docker-run upload-hf

# Default target
help:
	@echo "YOLO26 vs YOLO11 VisDrone Benchmark"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install dependencies"
	@echo "  install-dev    Install dev dependencies"
	@echo "  download-data  Download VisDrone dataset"
	@echo ""
	@echo "Training:"
	@echo "  train-yolo26   Train YOLO26n on VisDrone"
	@echo "  train-yolo11   Train YOLO11n on VisDrone"
	@echo "  train-all      Train both models"
	@echo ""
	@echo "Evaluation:"
	@echo "  validate       Validate trained models"
	@echo "  benchmark      Run full benchmark"
	@echo "  speed-test     Run speed benchmark"
	@echo "  coco-eval      Run COCO evaluation"
	@echo "  visualize      Generate comparison plots"
	@echo ""
	@echo "Development:"
	@echo "  format         Format code with black"
	@echo "  lint           Run linters"
	@echo "  test           Run tests"
	@echo "  clean          Clean generated files"
	@echo ""
	@echo "Deployment:"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo "  upload-hf      Upload model to HuggingFace"

# ==================== SETUP ====================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebook]"
	pre-commit install

download-data:
	python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').train(data='VisDrone.yaml', epochs=0)"
	@echo "VisDrone dataset downloaded to datasets/VisDrone"

# ==================== TRAINING ====================

# Training parameters
EPOCHS ?= 100
IMGSZ ?= 640
BATCH ?= 16
DEVICE ?= 0
PROJECT ?= runs/detect

train-yolo26:
	yolo detect train \
		model=yolo26n.pt \
		data=VisDrone.yaml \
		epochs=$(EPOCHS) \
		imgsz=$(IMGSZ) \
		batch=$(BATCH) \
		device=$(DEVICE) \
		project=$(PROJECT) \
		name=yolo26n_visdrone \
		exist_ok=True \
		plots=True \
		save=True

train-yolo11:
	yolo detect train \
		model=yolo11n.pt \
		data=VisDrone.yaml \
		epochs=$(EPOCHS) \
		imgsz=$(IMGSZ) \
		batch=$(BATCH) \
		device=$(DEVICE) \
		project=$(PROJECT) \
		name=yolo11n_visdrone \
		exist_ok=True \
		plots=True \
		save=True

train-all: train-yolo26 train-yolo11

# ==================== EVALUATION ====================

YOLO26_WEIGHTS ?= runs/detect/yolo26n_visdrone/weights/best.pt
YOLO11_WEIGHTS ?= runs/detect/yolo11n_visdrone/weights/best.pt

validate:
	@echo "Validating YOLO26n..."
	yolo detect val \
		model=$(YOLO26_WEIGHTS) \
		data=VisDrone.yaml \
		imgsz=$(IMGSZ) \
		batch=$(BATCH) \
		device=$(DEVICE) \
		project=$(PROJECT) \
		name=yolo26n_val \
		exist_ok=True
	@echo ""
	@echo "Validating YOLO11n..."
	yolo detect val \
		model=$(YOLO11_WEIGHTS) \
		data=VisDrone.yaml \
		imgsz=$(IMGSZ) \
		batch=$(BATCH) \
		device=$(DEVICE) \
		project=$(PROJECT) \
		name=yolo11n_val \
		exist_ok=True

benchmark:
	python -m src.evaluation.benchmark \
		--yolo26-weights $(YOLO26_WEIGHTS) \
		--yolo11-weights $(YOLO11_WEIGHTS) \
		--data VisDrone.yaml \
		--imgsz $(IMGSZ) \
		--device $(DEVICE)

speed-test:
	python -m src.evaluation.benchmark \
		--yolo26-weights $(YOLO26_WEIGHTS) \
		--yolo11-weights $(YOLO11_WEIGHTS) \
		--speed-only \
		--imgsz $(IMGSZ)

coco-eval:
	python -m src.evaluation.coco_eval \
		--yolo26-weights $(YOLO26_WEIGHTS) \
		--yolo11-weights $(YOLO11_WEIGHTS) \
		--data VisDrone.yaml

visualize:
	python -m src.utils.visualization \
		--results-dir $(PROJECT) \
		--output-dir assets

# ==================== DEVELOPMENT ====================

format:
	black src/ app/ tests/
	ruff check --fix src/ app/ tests/

lint:
	ruff check src/ app/ tests/
	black --check src/ app/ tests/
	mypy src/

test:
	pytest tests/ -v --cov=src --cov-report=html

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf build/ dist/ *.egg-info
	rm -rf runs/ outputs/ .hydra/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf datasets/

# ==================== DOCKER ====================

DOCKER_IMAGE ?= yolo26-visdrone
DOCKER_TAG ?= latest

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f docker/Dockerfile .

docker-run:
	docker run --gpus all -it --rm \
		-v $(PWD)/datasets:/app/datasets \
		-v $(PWD)/runs:/app/runs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

# ==================== HUGGINGFACE ====================

HF_REPO ?= raimbekovm/yolo26-visdrone
HF_TOKEN ?= $(shell echo $$HF_TOKEN)

upload-hf:
	python -c "\
	from huggingface_hub import HfApi; \
	api = HfApi(); \
	api.upload_file( \
		path_or_fileobj='$(YOLO26_WEIGHTS)', \
		path_in_repo='yolo26n_visdrone.pt', \
		repo_id='$(HF_REPO)', \
		repo_type='space', \
		token='$(HF_TOKEN)' \
	); \
	print('Model uploaded successfully!')"

# ==================== GRADIO ====================

demo:
	python app.py

demo-share:
	python app.py --share
