"""
YOLO26-VisDrone HuggingFace Spaces Demo.

This is the entry point for the HuggingFace Spaces demo application.
It provides a Gradio interface for object detection using YOLO26 trained on VisDrone.

Author: Murat Raimbekov
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.gradio_app import create_demo


def main():
    """Launch the Gradio demo."""
    import argparse

    parser = argparse.ArgumentParser(description="YOLO26-VisDrone Demo")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
