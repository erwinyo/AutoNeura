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
from app.text_recognition import TextRecognition
from utils.files import(
    is_media_file
)

load_dotenv()

def main(source_filepath: str, output_filename: str = "output.mp4"):
    logger.info("Starting the AutoNeura...")

    # Initialize TextRecognition
    logger.info("Initializing TextRecognition class.")
    text_recognition_app = TextRecognition(**text_recognition_user_config)
    
    # Check if the source file is a media file
    is_media, media_type = is_media_file(source_filepath)
    if not is_media:
        logger.error(f"File {source_filepath} is not a media file.")
        return
    if media_type == "video":
        logger.info("Processing video file.")
        pass
    elif media_type == "image":
        logger.info("Processing image file.")
        image = cv2.imread(source_filepath)
        if image is None:
            logger.error(f"Could not read image file {source_filepath}.")
            return

        # Perform OCR
        logger.info("Performing OCR.")  
        results = text_recognition_app.process(image)
        
        logger.trace("Results: ", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Recognition Recognition")
    parser.add_argument(
        "--source", 
        type=str, 
        required=True,
        help="Path to the input file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Name of the output file."
    )
    
    args = parser.parse_args()
    
    main(
        source_filepath=args.source, 
        output_filename=args.output
    )