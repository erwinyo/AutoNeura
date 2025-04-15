import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

import cv2
from rich import print
from tqdm import tqdm
import supervision as sv
from dotenv import load_dotenv

from base.yolo_vehicle_detection import YOLOVehicleDetection
from base.yolo_license_plate_detection import YOLOLicensePlateDetection 
from base.color_detection import ColorDetection
from base.doctr_ocr import DocTROcr
from base.config import (
    logger,
    yolo_vehicle_detection_user_config,
    doctr_ocr_user_config
)
load_dotenv()

def main(source_filepath: str):
    logger.info("Starting the AutoNeura...")

    # Initialize YOLO vehicle detection
    logger.info("Initializing YOLOVehicleDetection with user configuration.")
    yolo_vehicle_detection = YOLOVehicleDetection(
        config=yolo_vehicle_detection_user_config
    )
    # Initialize YOLO license plate detection
    logger.info("Initializing YOLOLicensePlateDetection with user configuration.")
    yolo_license_plate_detection = YOLOLicensePlateDetection(
        config=yolo_vehicle_detection_user_config
    )

    # Initialize annotation tools
    logger.info("Initializing annotation tools.")
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"])
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"]),
        text_color=sv.Color.BLACK,
        text_scale=0.35,
        text_padding=2
    )

    # Initialize doctr
    logger.info("Initializing DocTROcr with user configuration.")
    doctr_ocr = DocTROcr(
        config=doctr_ocr_user_config
    )


     # Load video
    logger.info("Initializing OpenCV VideoCapture.")
    cap = cv2.VideoCapture(source_filepath)
    # Initialize video writer
    logger.info("Initializing OpenCV VideoWriter.")
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize tqdm progress bar
    logger.info("Initializing tqdm progress bar.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    logger.info("Starting the main loop.")
    while True:
        logger.info("Reading frame from video.")

        has_frame, frame = cap.read()
        logger.debug(f"Frame read status: {has_frame}")
        logger.debug(f"Frame shape: {frame.shape}")

        if not has_frame:
            logger.error("No more frames to read, breaking the loop.")
            break

        # Perform vehicle detection on the frame
        logger.info("Performing vehicle detection.")
        vehicle_detections = yolo_vehicle_detection.process(frame)
        logger.debug(f"Vehicle detections atttributes: {dir(vehicle_detections)}")
        logger.trace(f"Vehicle detections xyxy: {vehicle_detections.xyxy}")
        logger.trace(f"Vehicle detections tracker_id: {vehicle_detections.tracker_id}")
        
        # Perform license plate detection on the frame
        logger.info("Performing license plate detection.")
        license_plate_detections = yolo_license_plate_detection.process(frame)
        logger.debug(f"License plate detections atttributes: {dir(license_plate_detections)}")
        logger.trace(f"License plate detections xyxy: {license_plate_detections.xyxy}")
        logger.trace(f"License plate detections tracker_id: {license_plate_detections.tracker_id}")
        



        # Annotate the frame with detected vehicles
        logger.info("Annotating the frame with detected vehicles.")
        annotated_frame = ellipse_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        logger.info("Annotating the frame with labels.")
        annotated_image = label_annotator.annotate(
            scene=annotated_image, 
            detections=detections, 
            labels=labels
        )
        
        logger.info("Drawing the labels on the frame.")
        out.write(annotated_frame)

        progress_bar.update(1)



def parse_arguments():
    parser = argparse.ArgumentParser(description="AutoNeura CLI")
    parser.add_argument(
        "--source",
        type=str,
        help="Path to the source (video/image file)",
        required=False
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