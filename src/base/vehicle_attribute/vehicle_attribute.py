# Built-in package
import os
from dataclasses import dataclass, field

# Third party package
from PIL import Image
import supervision as sv
from dotenv import load_dotenv
from paddleclas import PaddleClas

# Local package
from base.config import (
    logger,
    vehicle_attribute_model_config,
    vehicle_attribute_inference_config
)

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class VehicleAttribute:
    config: dict = field(default_factory=dict)

    _model: PaddleClas = field(init=False, repr=False)
    def __post_init__(self) -> None:
       logger.info("Initializing VehicleAttribute class.")
       self._model = PaddleClas(
            model_name="vehicle_attribute"
        )
    #    self._model = PaddleClas(**vehicle_attribute_model_config)
    def preprocess(self, data):
        return data

    def postprocess(self, attributes):
        detections = sv.Detections.from_paddledet(attributes)

        return detections

    def process(self, image, raw_result: bool = False):
        attributes = self._model.predict(
            image,
            **vehicle_attribute_inference_config
        )
        
        if raw_result:
            return attributes

        attributes = self.postprocess(attributes)
        return attributes