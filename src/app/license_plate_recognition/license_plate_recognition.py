# Built-in package
import os
import time
from dataclasses import dataclass, field

# Third party package
import supervision as sv
from dotenv import load_dotenv
import torchvision.transforms as T

# Local package
from base.config import (
    logger,
    yolo_vehicle_detection_user_config,
    yolo_license_plate_detection_user_config,
    doctr_ocr_user_config
)
from app.license_plate_recognition.helper import (
    is_license_plate_inside_vehicle
)
from base.yolo_vehicle_detection import YOLOVehicleDetection    
from base.yolo_license_plate_detection import YOLOLicensePlateDetection
from base.doctr_ocr import DocTROcr

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class LicensePlateRecognition:
    config: dict = field(default_factory=dict)

    _vehicle_detection_model: YOLOVehicleDetection = field(init=False, repr=False)
    _license_plate_detection_model: YOLOLicensePlateDetection = field(init=False, repr=False)
    _ocr_model: DocTROcr = field(init=False, repr=False) 
    _annotator: sv.EllipseAnnotator = field(init=False, repr=False)
    _label_annotator: sv.LabelAnnotator = field(init=False, repr=False)   
    def __post_init__(self) -> None:
        logger.info("Initializing LicensePlateRecognition class.")

        # Initialize YOLO vehicle detection
        self._vehicle_detection_model = YOLOVehicleDetection(   
            config=yolo_vehicle_detection_user_config
        )

        # Initialize YOLO license plate detection
        self._license_plate_detection_model = YOLOLicensePlateDetection(
            config=yolo_license_plate_detection_user_config
        )
        
        # Initialize doctr
        self._ocr_model = DocTROcr(
            config=doctr_ocr_user_config
        )

    def preprocess(self, data):
        return data

    def postprocess(self, data):
        return data

    def process(self, image, raw_result: bool = False):
        # Perform vehicle detection on the frame
        vehicle_detections = self._vehicle_detection_model.process(image)
        # logger.debug(f"Vehicle detections atttributes: {dir(vehicle_detections)}")
        # logger.trace(f"Vehicle detections xyxy: {vehicle_detections.xyxy}")
        # logger.trace(f"Vehicle detections tracker_id: {vehicle_detections.tracker_id}")

        # Perform license plate detection on the frame
        license_plate_detections = self._license_plate_detection_model.process(image)
        # logger.debug(f"License plate detections atttributes: {dir(license_plate_detections)}")
        # logger.trace(f"License plate detections xyxy: {license_plate_detections.xyxy}")
        # logger.trace(f"License plate detections tracker_id: {license_plate_detections.tracker_id}")

        # Check license plate detection is inside vehicle or not
        start = time.time()
        insides = is_license_plate_inside_vehicle(
            vehicle_detection=vehicle_detections,
            license_plate_detection=license_plate_detections
        )
        end = time.time()
        print(f"Time taken to check license plate inside vehicle: {end - start}")

        # Filter out license plate detections that are not inside vehicles
        start = time.time()
        filtered_indices = [i for i, inside in enumerate(insides) if inside]
        filtered_xyxy = license_plate_detections.xyxy[filtered_indices] if len(filtered_indices) > 0 else []
        # logger.trace(f"Filtered license plate detections xyxy: {filtered_xyxy}")
        end = time.time()
        print(f"Time taken to filter license plate detections: {end - start}")

        # Perform OCR on the filtered license plate detections
        start = time.time()
        license_plates = []
        for xyxy in filtered_xyxy:
            # Extract the license plate region from the image
            x1, y1, x2, y2 = map(int, xyxy)
            license_plate_region = image[y1:y2, x1:x2]

            # Perform OCR on the license plate region
            ocr_result = self._ocr_model.process(license_plate_region)
            logger.debug(f"OCR result: {ocr_result}")

            # Append the OCR result to the list of license plates
            license_plates.append(ocr_result)
        end = time.time()
        print(f"Time taken to perform OCR: {end - start}")


        if raw_result:
            return license_plates, vehicle_detections, license_plate_detections

        license_plates = self.postprocess(license_plates)
        return license_plates, vehicle_detections, license_plate_detections