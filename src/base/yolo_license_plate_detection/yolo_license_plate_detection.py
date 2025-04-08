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
    logger
)
from base.config import (
    yolo_license_plate_model_config,
    yolo_license_plate_inference_config
)

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class YOLOLicensePlateDetection:
    config: dict = field(default_factory=dict)

    _model: YOLO = field(init=False, repr=False)
    def __post_init__(self) -> None:
        logger.info("Initializing YOLOLicensePlateDetection class.")
        self._model = YOLO(**yolo_license_plate_model_config)

    def preprocess(self, data):
        logger.info("Preprocessing data for YOLOLicensePlateDetection class.")
        return data
    
    def postprocess(self, data):
        logger.info("Postprocessing data for YOLOLicensePlateDetection class.")
        return data

    def process(self, data, raw_result: bool = False, save_result: bool = False):
        logger.info("Processing data YOLOLicensePlateDetection class.")
        
        if save_result:
            if not os.path.exists(data):
                logger.error(f"The specified data path does not exist: {data}")
                raise FileNotFoundError(f"The specified data path does not exist: {data}")
            
            # Save result to file
            logger.info("Saving result to file.")
            self._model(data, save=True, **yolo_license_plate_inference_config)
            return

        else:
            with torch.no_grad():
                results = self._model.predict(data, **yolo_license_plate_inference_config)[0]

            # Compile result to supervision
            detections = sv.Detections.from_ultralytics(results)

            if raw_result:
                return detections

            detections = self.postprocess(detections)
            return detections