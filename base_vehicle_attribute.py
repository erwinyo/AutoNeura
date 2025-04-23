# Built-in package
import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

# Third party package
import cv2
from rich import print
from tqdm import tqdm
import supervision as sv
from dotenv import load_dotenv

# Local package
from base.config import (
    logger,
    vehicle_attribute_user_config
)
from base.vehicle_attribute import VehicleAttribute

load_dotenv()

def main():
    vehicle_attribute = VehicleAttribute(
        config=vehicle_attribute_user_config
    )

    


if __name__ == "__main__":
    main()