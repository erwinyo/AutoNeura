import os
import sys
from datetime import datetime

from box import Box
from loguru import logger

# Logger configuration
LEVEL = "TRACE"
# LEVEL = "DEBUG"

logger.remove() # Remove default logger configuration
# Add new logger configuration to write to a file
logger.add(
    f"logs/{LEVEL}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    format="<yellow>[{time:YYYY-MM-DD HH:mm:ss:SSSS}]</yellow> [<level><b>{level}</b></level>] [<b>{file.path}:{line}</b>] [<b>{function}</b>] <level>{message}</level>",
    level=LEVEL
)
# Add new logger configuration to print to console
logger.add(
    sys.stdout,
    format="<yellow>[{time:YYYY-MM-DD HH:mm:ss:SSSS}]</yellow> [<level><b>{level}</b></level>] [<b>{file.path}:{line}</b>] [<b>{function}</b>] <level>{message}</level>",
    level=LEVEL
)

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
    "device": "cuda:0",
    "agnostic_nms": False,
    "classes": [2, 3, 5, 7],
    "stream": False,
    "verbose": False
})

# Yolo License Plate configuration
yolo_license_plate_user_config = Box({

})
yolo_license_plate_model_config = Box({
    "model": "resources/models/yolo/license_plate/license_plate_yolo12n.pt",
    "task": "detect"
})
yolo_license_plate_inference_config = Box({
    "conf": 0.4,
    "iou": 0.7,
    "half": False,
    "device": "cuda:0",
    "agnostic_nms": False,
    "classes": [0],
    "stream": False,
    "verbose": False
})


# Color Detection configuration
color_detection_user_config = Box({
    "iscc_nbs_colour_system_path": "resources/files/iscc-nbs-colour-system.xlsx"
})
color_detection_model_config = Box({

})


# Vehicle Attribute configuration
vehicle_attribute_user_config = Box({
})
vehicle_attribute_model_config = Box({
    "model_name": "vehicle_attribute",
    "use_gpu": False,
    "batch_size": 1,
    "topk": 5,
})
vehicle_attribute_inference_config = Box({
    "predict_type": "cls"
})



# Doctr OCR configuration
doctr_ocr_user_config = Box({
    
})
doctr_ocr_model_config = Box({
    "det_arch": "db_resnet50",
    "reco_arch": "parseq",
    "pretrained": True
})