import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

import cv2
from tqdm import tqdm
import supervision as sv
from dotenv import load_dotenv

from base.yolo_vehicle_detection import YOLOVehicleDetection
from base.color_detection import ColorDetection
from base.doctr_ocr import DocTROcr
from base.config import (
    logger,
    yolo_vehicle_detection_user_config,
    color_detection_user_config,
    doctr_ocr_user_config
)
load_dotenv()

def main():
    logger.info("Starting the AutoNeura...")

    # Initialize color detection
    logger.info("Initializing ColorDetection with user configuration.")
    color_detection = ColorDetection(
        config=color_detection_user_config
    )

    # Initialize YOLO vehicle detection
    logger.info("Initializing YOLOVehicleDetection with user configuration.")
    yolo_vehicle_detection = YOLOVehicleDetection(
        config=yolo_vehicle_detection_user_config
    )

    # Initialize annotation tools
    logger.info("Initializing annotation tools.")
    logger.info("Initializing Ellipse Annotator.")
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"])
    )
    logger.info("Initializing Label Annotator.")
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"]),
        text_color=sv.Color.BLACK,
        text_scale=0.35,
        text_padding=2
    )

    # Load video
    logger.info("Initializing OpenCV VideoCapture.")
    cap = cv2.VideoCapture("/home/erwin/Videos/cars/sample#7.mp4")
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
        detections = yolo_vehicle_detection.process(frame)
        logger.debug(f"Detections atttributes: {dir(detections)}")
        logger.debug(f"Detections xyxy: {detections.xyxy}")  

        logger.info("Performing color detection.")
        dominant_hue_colors = []
        dominant_hue_categories = []

        logger.info("Processing each detection with for loop.")
        for x1, y1, x2, y2 in detections.xyxy:
            logger.debug(f"Detection coordinates: {x1}, {y1}, {x2}, {y2}")
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            logger.debug(f"Converted coordinates: {x1}, {y1}, {x2}, {y2}")
            mid_x, mid_y = int(x1 + ((x2 - x1) / 2)), int(y1 + ((y2 - y1) / 2))
            logger.debug(f"Midpoint coordinates: {mid_x}, {mid_y}") 

            cropped_frame = frame[y1:y2, x1:x2]
            logger.debug(f"Cropped frame shape: {cropped_frame.shape}")

            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            logger.debug(f"Resized cropped frame shape: {cropped_frame.shape}")

            dominant_hue_color, dominant_hue_category = color_detection.process(cropped_frame)
            logger.debug(f"Dominant hue color: {dominant_hue_color}")
            logger.debug(f"Dominant hue category: {dominant_hue_category}")

            dominant_hue_colors.append(dominant_hue_color)
            dominant_hue_categories.append(dominant_hue_category)
        
        else:
            logger.info("[loop detections.xyxy] For loop done. No detections found in the frame.")
        
        logger.debug(f"Dominant hue colors: {len(dominant_hue_colors)}")
        logger.debug(f"Dominant hue categories: {len(dominant_hue_categories)}")

        logger.info("Creating labels for detected vehicles.")
        labels = [
            f"Color: {color}, Category: {category}"
            for color, category 
            in zip(dominant_hue_colors, dominant_hue_categories)
        ]

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

    logger.info("Closing video writer and releasing resources.")
    progress_bar.close()
    cap.release()
    out.release()


def main_license_plate():
    logger.info("Starting the AutoNeura...")

    # Initialize DocTROcr
    logger.info("Initializing DocTROcr with user configuration.")
    doctr_ocr = DocTROcr(
        config=doctr_ocr_user_config
    )

    # Load image
    logger.info("Loading image.")
    image = cv2.imread("/home/erwin/Images/brazil-plate2.jpg")
    logger.debug(f"Image shape: {image.shape}")

    # Detect license plate
    logger.info("Detecting license plate.")
    results = doctr_ocr.process(image)
    logger.debug(f"Results: {results}")

if __name__ == "__main__":
    main_license_plate()
        


