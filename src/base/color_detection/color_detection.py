# Built-in package
import os
from dataclasses import dataclass, field

# Third party package
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from dotenv import load_dotenv

# Local package
from base.config import (
    logger
)
from base.color_detection.helper import (
    calculate_manhattan_distance,
    calculate_chebyshev_distance,
    calculate_minkowski_distance,
    calculate_euclidean_distance
)
load_dotenv()

@dataclass(frozen=False, kw_only=False, match_args=False, slots=False)
class ColorDetection:
    config: dict = field(default_factory=dict)

    _color_system: pd.DataFrame = field(init=False, repr=False)
    _iscc_color: np.ndarray = field(init=False, repr=False)
    _iscc_category: np.ndarray = field(init=False, repr=False)
    _iscc_rgb: np.ndarray = field(init=False, repr=False)
    def __post_init__(self) -> None:
        logger.info("Initializing ColorDetection class.")

        # Setup the color system
        self._color_system = pd.read_excel(
            self.config.iscc_nbs_colour_system_path
        ).dropna(subset=['r', 'g', 'b']).reset_index(drop=True)
        logger.trace(f"Raw color system data from file: {self._color_system.head()}")

        # Seperate the color and category
        self._iscc_color = self._color_system[["color"]].values
        self._iscc_category = self._color_system[["category"]].values

        # Convert RGB to LAB color space
        self._iscc_rgb = self._color_system[['r', 'g', 'b']].to_numpy() / 255
        self._iscc_lab = color.rgb2lab(self._iscc_rgb)

    def preprocess(self, data):
        return data

    def postprocess(self, dominant_hue_color, dominant_hue_category):
        return dominant_hue_color, dominant_hue_category

    def process(self, rgb, metric: str = "euclidean", raw_result: bool = False):
        logger.debug(f"Type of rgb variable: {type(rgb)}")
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb)
            logger.debug(f"Type of rgb variable: {type(rgb)}")
        
        logger.debug(f"Shape of rgb variable: {rgb.shape}")
        lab = color.rgb2lab(rgb / 255)
        logger.debug(f"Shape of lab variable: {lab.shape}")

        # Find the closest color in LUT1
        t = []
        for i in range(len(self._iscc_lab)):
            if metric == "euclidean":
                distances = calculate_euclidean_distance(
                    array1=self._iscc_lab[i], 
                    array2=lab
                )
                logger.debug(f"Euclidean distance: {distances}")
            elif metric == "manhattan":
                distances = calculate_manhattan_distance(
                    vector1=self._iscc_lab[i], 
                    vector2=lab
                )
                logger.debug("Manhattan distance: {distances}")
            elif metric == "chebyshev":
                distances = calculate_chebyshev_distance(
                    vector1=self._iscc_lab[i], 
                    vector2=lab
                )
                logger.debug("Chebyshev distance: {distances}")
            elif metric == "minkowski":
                distances = calculate_minkowski_distance(
                    vector1=self._iscc_lab[i], 
                    vector2=lab, 
                    power_parameter=3
                )
                logger.debug("Minkowski distance: {distances}")
            t.append(distances)
        
        distances = np.array(t)
        closest_color_index = np.argmin(distances)
        logger.debug(f"Closest color index: {closest_color_index}")

        # Get the dominant hue from LUT1
        dominant_hue_color = str(self._iscc_color[closest_color_index][0])
        logger.debug(f"Dominant hue color: {dominant_hue_color}")
        dominant_hue_category = str(self._iscc_category[closest_color_index][0])
        logger.debug(f"Dominant hue category: {dominant_hue_category}")

        if raw_result:
             return dominant_hue_color, dominant_hue_category

        dominant_hue_color, dominant_hue_category = self.postprocess(
            dominant_hue_color, 
            dominant_hue_category
        )
        return dominant_hue_color, dominant_hue_category

