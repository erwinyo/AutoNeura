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
    yolo_license_plate_detection_model_config,
    yolo_license_plate_detection_inference_config,
)
from base.yolo_vehicle_detection import YoloVehicleDetection

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class VehicleCounter:
    config: dict = field(default_factory=dict)

    _model: YoloVehicleDetection = field(init=False, repr=False)
    def __post_init__(self) -> None:
       logger.info("Initializing VehicleCounter class.")
       self._model = YoloVehicleDetection(
            model_name="vehicle_counter"
        )

    def preprocess(self, data):
        logger.info("Preprocessing data for VehicleCounter class.")
        return data

    def postprocess(self, data):
        logger.info("Postprocessing data for VehicleCounter class.")
        return data

    def process(self, image, raw_result: bool = False):
        logger.info("Processing data VehicleCounter class.")


        
        
        if raw_result:
            return embedding

        embedding = self.postprocess(embedding)
        return embedding