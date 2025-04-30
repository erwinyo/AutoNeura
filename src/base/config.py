import os
import sys
from datetime import datetime

from box import Box
from loguru import logger

# Logger configuration
LEVEL = "TRACE"
PRINT_TO_CONSOLE = True

# ------------------------------- [LOGGER] -------------------------------

logger.remove() # Remove default logger configuration
# Add new logger configuration to write to a file
logger.add(
    f"logs/{LEVEL}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    format="<yellow>[{time:YYYY-MM-DD HH:mm:ss:SSSS}]</yellow> [<level><b>{level}</b></level>] [<b>{file.path}:{line}</b>] [<b>{function}</b>] <level>{message}</level>",
    level=LEVEL
)

if PRINT_TO_CONSOLE:
    # Add new logger configuration to print to console
    logger.add(
        sys.stdout,
        format="<yellow>[{time:YYYY-MM-DD HH:mm:ss:SSSS}]</yellow> [<level><b>{level}</b></level>] [<b>{file.path}:{line}</b>] [<b>{function}</b>] <level>{message}</level>",
        level=LEVEL
    )

# ------------------------------- [APP] -------------------------------
license_plate_recognition_user_config = Box({  

})



# ------------------------------- [BASE] -------------------------------
# Yolo Vehicle Detection configuration
yolo_vehicle_detection_user_config = Box({

})
yolo_vehicle_detection_model_config = Box({
    "model": "resources/models/yolo/original/yolo12m.pt",
    "task": "detect"
})
yolo_vehicle_detection_inference_config = Box({
    "conf": 0.4,
    "iou": 0.7,
    "half": False,
    "device": "0",
    "agnostic_nms": False,
    "classes": [2, 3, 5, 7],
    "stream": False,
    "verbose": True
})

# Yolo License Plate configuration
yolo_license_plate_detection_user_config = Box({

})
yolo_license_plate_detection_model_config = Box({
    "model": "resources/models/yolo/license_plate/license_plate_yolo12n.pt",
    "task": "detect"
})
yolo_license_plate_detection_inference_config = Box({
    "conf": 0.4,
    "iou": 0.7,
    "half": False,
    "device": "0",
    "agnostic_nms": False,
    "classes": [0],
    "stream": False,
    "verbose": True
})


# Color Detection configuration
color_detection_user_config = Box({
    "iscc_nbs_colour_system_path": "resources/files/iscc-nbs-colour-system.xlsx"
})
color_detection_model_config = Box({

})


# Doctr OCR configuration
doctr_ocr_user_config = Box({
    "use_gpu": True,
    "use_half_precision": True,
})
doctr_ocr_model_config = Box({
    "det_arch": "db_resnet50",
    "reco_arch": "parseq",
    "pretrained": True,
    "assume_straight_pages": False,
})