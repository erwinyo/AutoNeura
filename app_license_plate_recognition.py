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
    logger.info("Starting the License Plate Recognition App")

    # Initialize LicensePlateRecognition
    license_plate_recognition_app = LicensePlateRecognition(
        config=license_plate_recognition_user_config
    )

    # Initialize annotation tools
    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"])
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"]),
        text_color=sv.Color.BLACK,
        text_scale=0.35,
        text_padding=2
    )

     # Load video
    cap = cv2.VideoCapture(source_filepath)
    # Initialize video writer
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    while True:
        has_frame, frame = cap.read()
        
        if not has_frame:
            logger.error("No more frames to read, breaking the loop.")
            break

        license_plates, vehicle_detections, license_detections = license_plate_recognition_app.process(
            image=frame,
            raw_result=False
        )
        logger.trace(f"License Plates: {license_plates}")

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=vehicle_detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=license_detections
        )
        
        out.write(annotated_frame)
        progress_bar.update(1)


if __name__ == "__main__":
    main(
        source_filepath="/home/erwin/Documents/AutoNeura/resources/videos/cars/sample#1.mp4",
        output_filename="output.mp4"
    )