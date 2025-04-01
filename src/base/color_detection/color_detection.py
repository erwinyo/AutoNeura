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
        # Setup the color system
        self._color_system = pd.read_excel(
            self.config.iscc_nbs_colour_system_path
        ).dropna(subset=['r', 'g', 'b']).reset_index(drop=True, inplace=True)

        self._iscc_color = self.color_system[["color"]].values
        self._iscc_category = self.color_system[["category"]].values
        self._iscc_rgb = self.color_system[['r', 'g', 'b']].values / 255
        self._iscc_lab = color.rgb2lab(self._iscc_rgb)

    def preprocess(self, data):
        return data

    def postprocess(self, dominant_hue_color, dominant_hue_category):
        return dominant_hue_color, dominant_hue_category

    def process(self, rgb, metric: str = "euclidean", raw_result: bool = False):
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb)
        
        rgb = rgb / 255
        lab = color.rgb2lab(rgb)

        # Find the closest color in LUT1
        t = []
        for i in range(len(self._iscc_lab)):
            if metric == "euclidean":
                distances = calculate_euclidean_distance(self._iscc_lab[i], lab)
            elif metric == "manhattan":
                distances = calculate_manhattan_distance(self._iscc_lab[i], lab)
            elif metric == "chebyshev":
                distances = calculate_chebyshev_distance(self._iscc_lab[i], lab)
            elif metric == "minkowski":
                distances = calculate_minkowski_distance(self._iscc_lab[i], lab, 3)
            t.append(distances)
        distances = np.array(t)
        closest_color_index = np.argmin(distances)

        # Get the dominant hue from LUT1
        dominant_hue_color = str(self._iscc_color[closest_color_index][0])
        dominant_hue_category = str(self._iscc_category[closest_color_index][0])

        if raw_result:
             return dominant_hue_color, dominant_hue_category

        dominant_hue_color, dominant_hue_category = self.postprocess(
            dominant_hue_color, 
            dominant_hue_category
        )
        return dominant_hue_color, dominant_hue_category
