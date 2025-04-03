# Built-in package
import os
from dataclasses import dataclass, field

# Third party package
import torch
from paddleclas import PaddleClas
from PIL import Image
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv
import torchvision.transforms as T
from transformers import AutoFeatureExtractor, AutoModel

# Local package
from base.config import (
    logger
)

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class VehicleAttribute:
    config: dict = field(default_factory=dict)

    _model: PaddleClas = field(init=False, repr=False)
    def __post_init__(self) -> None:
       logger.info("Initializing VehicleAttribute class.")
       self._model = PaddleClas(model_name="vehicle_attribute")

    def preprocess(self, data):
        logger.info("Preprocessing data for VehicleAttribute class.")
        return data

    def postprocess(self, attributes):
        logger.info("Postprocessing data for VehicleAttribute class.")
        return attributes

    def process(self, image, raw_result: bool = False):
        logger.info("Processing data VehicleAttribute class.")

        # Get the attribute of vehicle
        logger.info("Getting the attribute of vehicle.")
        attributes = self._model.predict(
            image,
            predict_type="cls"
        )
        
        if raw_result:
            return attributes

        attributes = self.postprocess(attributes)
        return attributes