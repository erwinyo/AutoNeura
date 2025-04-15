# Built-in package
import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

# Third party package
import cv2
from rich import print
from tqdm import tqdm
import supervision as sv
from dotenv import load_dotenv

# Local package
from base.config import (
    logger,
    license_plate_recognition_user_config
)
from app.license_plate_recognition import LicensePlateRecognition

load_dotenv()

def main(source_filepath: str, output_filename: str = "output.mp4"):
    logger.info("Starting the AutoNeura...")

    # Initialize LicensePlateRecognition
    logger.info("Initializing LicensePlateRecognition class.")
    license_plate_recognition_app = LicensePlateRecognition(
        **license_plate_recognition_user_config
    )

    # Initialize annotation tools
    logger.info("Initializing annotation tools.")
    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"])
    )
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"])
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"]),
        text_color=sv.Color.BLACK,
        text_scale=0.35,
        text_padding=2
    )

     # Load video
    logger.info("Initializing OpenCV VideoCapture.")
    cap = cv2.VideoCapture(source_filepath)
    # Initialize video writer
    logger.info("Initializing OpenCV VideoWriter.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        output_filename, 
        fourcc, 
        fps, 
        (frame_width, frame_height)
    )
    
    # Initialize tqdm progress bar
    logger.info("Initializing tqdm progress bar.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    logger.info("Starting the main loop.")
    while True:
        logger.info("Reading frame from video.")

        has_frame, frame = cap.read()
        logger.debug(f"Frame read status: {has_frame}")
        
        if not has_frame:
            logger.error("No more frames to read, breaking the loop.")
            break
        logger.debug(f"Frame shape: {frame.shape}")

        logger.info("Performing vehicle detection.")
        license_plates, vehicle_detections, license_detections = license_plate_recognition_app.process(
            image=frame,
            raw_result=False
        )

        logger.info("Annotating the frame with detected vehicles.")
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=vehicle_detections
        )
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=license_detections
        )
        
        out.write(annotated_frame)
        progress_bar.update(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoNeura CLI")
    parser.add_argument(
        "--source",
        type=str,
        help="Path to the source (video/image file)",
        required=True
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    logger.debug(f"Arguments entered by user: {args}")
    if args.source:
        logger.debug(f"Source file: {args.source}")
        if not os.path.exists(args.source):
            logger.error(f"The specified source file does not exist: {args.source}")
            raise FileNotFoundError(f"The specified source file does not exist: {args.source}")
        else:
            logger.info(f"Source file exists: {args.source}")
    else:
        logger.error("No source file provided.")
        raise ValueError("No source file provided.")
    
    main(
        source_filepath=args.source
    )