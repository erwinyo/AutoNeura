# Built-in package
import os
from dataclasses import dataclass, field

# Third party package
import torch
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv

# Local package
from base.config import (
    yolo_vehicle_detection_model_config,
    yolo_vehicle_detection_inference_config
)

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class YOLOVehicleDetection:
    config: dict = field(default_factory=dict)

    _model: YOLO = field(init=False, repr=False)
    def __post_init__(self) -> None:
        self._model = YOLO(**yolo_vehicle_detection_model_config)

    @staticmethod
    def preprocess(data):
        return data
    
    @staticmethod
    def postprocess(data):
        return data

    def process(self, image, raw_result: bool = False):
        with torch.no_grad():
            results = self._model.predict(image, **yolo_vehicle_detection_inference_config)[0]

        # Compile result to supervision
        detections = sv.Detections.from_ultralytics(results)

        if raw_result:
            return detections

        detections = self.postprocess(detections)
        return detections