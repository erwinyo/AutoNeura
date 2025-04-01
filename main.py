import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

import cv2
from tqdm import tqdm
import supervision as sv
from dotenv import load_dotenv

from base.yolo_vehicle_detection import YOLOVehicleDetection
from base.color_detection import ColorDetection
from base.config import (
    yolo_vehicle_detection_user_config,
    color_detection_user_config
)
load_dotenv()

if __name__ == "__main__":
    yolo_vehicle_detection = YOLOVehicleDetection(**yolo_vehicle_detection_user_config)

    # Annotation
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
    cap = cv2.VideoCapture("/home/erwin/Videos/cars/sample#7.mp4")
    # Initialize video writer
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize tqdm progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            print("Cannot read frame. Exiting...")
            break

        # Perform vehicle detection on the frame
        detections = yolo_vehicle_detection.process(frame)
        for x1, y1, x2, y2 in detections.xyxy:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mid_x, mid_y = int(x1 + ((x2 - x1) / 2)), int(y1 + ((y2 - y1) / 2))

            car = frame[y1:y2, x1:x2]
            car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)

            # Attribute analysis
            color_detection = ColorDetection(**color_detection_user_config)


        # Annotate the frame with detected vehicles
        annotated_frame = ellipse_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        
        out.write(annotated_frame)
        progress_bar.update(1)

    # Close video writer and release resources
    progress_bar.close()
    cap.release()
    out.release()
        


