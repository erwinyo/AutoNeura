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
    doctr_ocr_user_config
)
from base.doctr_ocr import DocTROcr

load_dotenv()

def main():
    doctr = DocTROcr(
        config=doctr_ocr_user_config
    )


    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"])
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"]),
        text_color=sv.Color.BLACK,
        text_scale=0.35,
        text_padding=2
    )



    img = cv2.imread("/home/erwin/Documents/AutoNeura/resources/images/license_plate/brazil-plate2.jpg")
    
    detections = doctr.process(img)
    print(f"Detections: {detections}")
    annotated_image = box_annotator.annotate(
        scene=img.copy(), 
        detections=detections
    )

    cv2.imshow("Image", annotated_image)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    

    

if __name__ == "__main__":
    main()