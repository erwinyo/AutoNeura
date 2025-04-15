# Built-in package
import os
from dataclasses import dataclass, field

# Third party package
import torch
from PIL import Image
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModel

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
        logger.info("Initializing YOLOVehicleDetection with user configuration.")
        self._vehicle_detection_model = YOLOVehicleDetection(   
            config=yolo_vehicle_detection_user_config
        )

        # Initialize YOLO license plate detection
        logger.info("Initializing YOLOLicensePlateDetection with user configuration.")
        self._license_plate_detection_model = YOLOLicensePlateDetection(
            config=yolo_license_plate_detection_user_config
        )
        
        # Initialize doctr
        logger.info("Initializing DocTROcr with user configuration.")
        self._ocr_model = DocTROcr(
            config=doctr_ocr_user_config
        )

        # Initialize annotation tools
        logger.info("Initializing annotation tools.")
        self._annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(["#FFFFFF"])
        )
        self._label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(["#FFFFFF"]),
            text_color=sv.Color.BLACK,
            text_scale=0.35,
            text_padding=2
        )

    def preprocess(self, data):
        logger.info("Preprocessing data for LicensePlateRecognition class.")
        return data

    def postprocess(self, data):
        logger.info("Postprocessing data for LicensePlateRecognition class.")
        return data

    def process(self, image, raw_result: bool = False):
        logger.info("Processing data LicensePlateRecognition class.")

        # Perform vehicle detection on the frame
        logger.info("Performing vehicle detection.")
        vehicle_detections = self._vehicle_detection_model.process(image)
        logger.debug(f"Vehicle detections atttributes: {dir(vehicle_detections)}")
        logger.trace(f"Vehicle detections xyxy: {vehicle_detections.xyxy}")
        logger.trace(f"Vehicle detections tracker_id: {vehicle_detections.tracker_id}")

        # Perform license plate detection on the frame
        logger.info("Performing license plate detection.")
        license_plate_detections = self._license_plate_detection_model.process(image)
        logger.debug(f"License plate detections atttributes: {dir(license_plate_detections)}")
        logger.trace(f"License plate detections xyxy: {license_plate_detections.xyxy}")
        logger.trace(f"License plate detections tracker_id: {license_plate_detections.tracker_id}")

        # Check license plate detection is inside vehicle or not
        logger.info("Performing license plate checking inside vehicle detections.")
        insides = is_license_plate_inside_vehicle(
            vehicle_detections_results=vehicle_detections,
            license_plate_detection_results=license_plate_detections
        )

        # Filter out license plate detections that are not inside vehicles
        logger.info("Performing filtering the license plate detections.")
        filtered_indices = [i for i, inside in enumerate(insides) if inside]
        filtered_license_plate_detections = license_plate_detections[filtered_indices]
        filtered_xyxy = license_plate_detections.xyxy[filtered_indices] if len(filtered_indices) > 0 else []

        
        




        if raw_result:
            return embedding

        embedding = self.postprocess(embedding)
        return embedding