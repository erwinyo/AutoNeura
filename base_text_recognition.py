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
    text_recognition_user_config
)
from base.doctr_ocr import DocTROcr

load_dotenv()

def main():
    doctr = DocTROcr(
        config=text_recognition_user_config
    )
    img = cv2.imread("/home/erwin/Documents/AutoNeura/resources/images/document/document2.png")
    
    result = doctr.process(img)
    print(result)

if __name__ == "__main__":
    main()