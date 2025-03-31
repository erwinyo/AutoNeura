import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

import cv2
from dotenv import load_dotenv

from base.yolo_vehicle_detection import YOLOVehicleDetection
from base.config import (
    yolo_vehicle_detection_user_config
)

load_dotenv()
if __name__ == "__main__":
    yolo_vehicle_detection = YOLOVehicleDetection(**yolo_vehicle_detection_user_config)

    cap = cv2.VideoCapture("/home/erwin/Videos/highway_demo.mp4")
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            print("Cannot read frame. Exiting...")
            break

        # # Perform vehicle detection on the frame
        # detections = yolo_vehicle_detection.detect(frame)

        # # Draw detections on the frame
        # for detection in detections:
        #     x, y, w, h = detection['bbox']
        #     label = detection['label']
        #     confidence = detection['confidence']
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Vehicle Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
        


