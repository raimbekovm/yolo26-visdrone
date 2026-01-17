"""
VisDrone Dataset Download and Verification.

This module handles downloading and verifying the VisDrone dataset.
Ultralytics provides built-in support for VisDrone, so we leverage that.
"""

import os
from pathlib import Path

from ultralytics import YOLO


def download_visdrone(data_dir: str | Path | None = None) -> Path:
    """
    Download VisDrone dataset using Ultralytics built-in support.

    Args:
        data_dir: Optional directory to store dataset. Defaults to ./datasets

    Returns:
        Path to the downloaded dataset directory
    """
    if data_dir is None:
        data_dir = Path("datasets")
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Use Ultralytics to download VisDrone
    # The simplest way is to start a training with epochs=0
    print("Downloading VisDrone dataset via Ultralytics...")
    print("This may take a few minutes depending on your connection.")

    try:
        # Initialize a model and trigger dataset download
        model = YOLO("yolo11n.pt")
        # epochs=0 will just download the dataset without training
        model.train(data="VisDrone.yaml", epochs=0, exist_ok=True)
    except Exception as e:
        # Training with 0 epochs might raise an error, but dataset will be downloaded
        if "VisDrone" not in str(e):
            print(f"Note: {e}")

    # Verify download
    visdrone_path = data_dir / "VisDrone"
    if visdrone_path.exists():
        print(f"VisDrone dataset downloaded to: {visdrone_path}")
        return visdrone_path
    else:
        # Check default Ultralytics location
        default_path = Path.home() / "datasets" / "VisDrone"
        if default_path.exists():
            print(f"VisDrone dataset found at: {default_path}")
            return default_path

    raise RuntimeError("Failed to download VisDrone dataset")


def verify_dataset(data_path: str | Path) -> dict:
    """
    Verify VisDrone dataset integrity.

    Args:
        data_path: Path to VisDrone dataset root

    Returns:
        Dictionary with verification results
    """
    data_path = Path(data_path)

    expected_splits = {
        "VisDrone2019-DET-train": {"images": 6471, "labels": 6471},
        "VisDrone2019-DET-val": {"images": 548, "labels": 548},
        "VisDrone2019-DET-test-dev": {"images": 1610, "labels": 1610},
    }

    results = {
        "valid": True,
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
    }

    for split_name, expected in expected_splits.items():
        split_path = data_path / split_name
        images_path = split_path / "images"
        labels_path = split_path / "labels"

        split_result = {
            "exists": split_path.exists(),
            "images_count": 0,
            "labels_count": 0,
            "images_valid": False,
            "labels_valid": False,
        }

        if images_path.exists():
            images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
            split_result["images_count"] = len(images)
            split_result["images_valid"] = len(images) >= expected["images"] * 0.95

        if labels_path.exists():
            labels = list(labels_path.glob("*.txt"))
            split_result["labels_count"] = len(labels)
            split_result["labels_valid"] = len(labels) >= expected["labels"] * 0.95

        results["splits"][split_name] = split_result
        results["total_images"] += split_result["images_count"]
        results["total_labels"] += split_result["labels_count"]

        if not (split_result["images_valid"] and split_result["labels_valid"]):
            results["valid"] = False

    return results


def print_dataset_info(data_path: str | Path) -> None:
    """Print VisDrone dataset information."""
    results = verify_dataset(data_path)

    print("\n" + "=" * 50)
    print("VisDrone Dataset Information")
    print("=" * 50)

    for split_name, split_info in results["splits"].items():
        status = "OK" if split_info["images_valid"] and split_info["labels_valid"] else "MISSING"
        print(f"\n{split_name}: [{status}]")
        print(f"  Images: {split_info['images_count']}")
        print(f"  Labels: {split_info['labels_count']}")

    print(f"\nTotal Images: {results['total_images']}")
    print(f"Total Labels: {results['total_labels']}")
    print(f"Dataset Valid: {results['valid']}")
    print("=" * 50)


def main():
    """CLI entry point for dataset download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download and verify VisDrone dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Directory to store dataset (default: datasets)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset without downloading",
    )
    args = parser.parse_args()

    if args.verify_only:
        print_dataset_info(args.data_dir)
    else:
        data_path = download_visdrone(args.data_dir)
        print_dataset_info(data_path)


if __name__ == "__main__":
    main()
