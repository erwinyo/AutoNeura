# Built-in package
import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

# Third party package
import numpy as np
from box import Box
from rich.style import Style
from rich.console import Console
from base.color_detection import ColorDetection

# Local package
from base.config import (
    logger,
    color_detection_user_config,
)

console = Console()

def generate_random_rgb_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return (r, g, b)

def generate_random_rgb_colors(n):
    colors = []
    for _ in range(n):
        color = generate_random_rgb_color()
        colors.append(color)
    return colors

def main():
    color_detection = ColorDetection(
        config=color_detection_user_config,
    )   
    colors = generate_random_rgb_colors(10)

    for color in colors:
        my_style = Style(color=f"rgb{color}")
        console.print(f"TEST COLOR!", style=my_style)

        color = np.array(color)
        print(f"Color: {color}")
        
        detection = color_detection.process(
            rgb=color,
            metric="euclidean"
        )
        print(f"Detection: {detection}")
        # print(f"Dominant hue color: {dominant_hue_color}")
        # print(f"Dominant hue category: {dominant_hue_category}")

if __name__ == "__main__":
    main()