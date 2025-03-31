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


load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class VITImageEmbedding:
    config: dict = field(default_factory=dict)

    _model: AutoFeatureExtractor = field(init=False, repr=False)
    def __post_init__(self) -> None:
       pass

    def preprocess(self, data):
        return data

    def postprocess(self, data):
        return data

    def process(self, image, raw_result: bool = False):
    
        
        if raw_result:
            return embedding

        embedding = self.postprocess(embedding)
        return embedding