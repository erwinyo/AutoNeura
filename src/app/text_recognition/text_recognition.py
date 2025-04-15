# Built-in package
import os
from dataclasses import dataclass, field

# Third party package
import supervision as sv
from dotenv import load_dotenv
import torchvision.transforms as T

# Local package
from base.config import (
    logger,
    text_recognition_user_config
)
from base.doctr_ocr import DocTROcr

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class TextRecognition:
    config: dict = field(default_factory=dict)

    _ocr_model: DocTROcr = field(init=False, repr=False)
    def __post_init__(self) -> None:
       logger.info("Initializing TextRecognition class.")
       
       logger.info("Initializing DocTROcr with user configuration.")
       self._ocr_model = DocTROcr(**text_recognition_user_config)    

    def preprocess(self, data):
        logger.info("Preprocessing data for TextRecognition class.")
        return data

    def postprocess(self, data):
        logger.info("Postprocessing data for TextRecognition class.")
        return data

    def process(self, image, raw_result: bool = False):
        logger.info("Processing data TextRecognition class.")
        
        results = self._ocr_model.process(
            image,
            raw_result=False
        )

        if raw_result:
            return results

        results = self.postprocess(results)
        return results