# Built-in package
import os
from dataclasses import dataclass, field

# Third party package
import torch
import numpy as np
from box import Box
import supervision as sv
from dotenv import load_dotenv
from doctr.io import DocumentFile
from doctr.models import (
    ocr_predictor, 
    kie_predictor
)
# Local package
from base.doctr_ocr.helper import (
    do_ocr_on_pdf,
    do_ocr_on_image
)
from base.config import (
    logger,
    doctr_ocr_model_config
)

load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class DocTROcr:
    config: dict = field(default_factory=dict)

    _model: ocr_predictor = field(init=False, repr=False)
    def __post_init__(self) -> None:
        logger.info("Initializing DocTROcr class.")

        self._model = ocr_predictor(**doctr_ocr_model_config)
        if self.config.use_gpu:
            self._model = self._model.cuda()
            logger.info("Using GPU for DocTROcr model.")
        if self.config.use_half_precision and self.config.use_gpu:
            self._model = self._model.half()
            logger.info("Using half precision for DocTROcr model.")
        else:
            logger.warning("Half precision is not supported on CPU. Please use GPU for half precision. Used CPU instead.")
    
    def preprocess(self, data):
        return data

    def postprocess(self, result, width_height):
        result_as_dict = Box(result.export())
        logger.trace(f"Raw result (JSON) of the DocTROcr: {result_as_dict}")

        words_bowl = []
        confidences_bowl = []
        objectness_score_bowl = [] # Determine the confidence score for text detection
        xyxys_bowl = []

        for page in result_as_dict.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        words_bowl.append(word.value)
                        confidences_bowl.append(word.confidence)
                        objectness_score_bowl.append(word.objectness_score)
                        xyxys_bowl.append(sum([[float(value) for value in row] for row in word.geometry], []))

        # Convert percentage coordinate to pixel coordinate
        width, height = width_height
        for i in range(len(xyxys_bowl)):
            xyxys_bowl[i] = [xyxys_bowl[i][0] * width, xyxys_bowl[i][1] * height, xyxys_bowl[i][2] * width, xyxys_bowl[i][3] * height]
        
        # Return the result as a dictionary
        return_value = Box({
            "words": words_bowl,
            "confidences": confidences_bowl,
            "objectness_score": objectness_score_bowl, 
            "xyxys": xyxys_bowl
        })
        return return_value

    def process(self, document, raw_result: bool = False):

        width, height = document.shape[:2]  
        result = self._model([document])
        logger.trace(f"Raw result of the DocTROcr: {result}")
        if raw_result:
            return result
        result = self.postprocess(result, (width, height))
        return result