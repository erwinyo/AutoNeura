# Built-in package
import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

# Third party package
import cv2
from tqdm import tqdm
from rich import print
from tqdm import tqdm
import supervision as sv
from dotenv import load_dotenv

# Local package
from base.config import (
    logger,
    yolo_vehicle_detection_user_config
)
from base.yolo_vehicle_detection import YOLOVehicleDetection

load_dotenv()

def main():
    model = YOLOVehicleDetection(
        config=yolo_vehicle_detection_user_config
    )
    capture = cv2.VideoCapture("/home/erwin/Documents/AutoNeura/resources/videos/cars/sample#1.mp4")

    # Get the total frame count for the progress bar
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (int(capture.get(3)), int(capture.get(4)))
    )

    # Wrap the loop with tqdm for a progress bar
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        has_frame, frame = capture.read()
        if not has_frame:
            break
        results = model.process(frame)
        print(results)
        
        # Write the processed frame to the output video
        out.write(frame)

    # Release the video objects
    capture.release()
    out.release()


if __name__ == "__main__":
    main()