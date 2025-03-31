from box import Box

yolo_vehicle_detection_user_config = Box({

})

yolo_vehicle_detection_model_config = Box({
    "model": "models/yolo/yolo12m.pt",
    "task": "detect"
})

yolo_vehicle_detection_inference_config = Box({
    "conf": 0.85,
    "iou": 0.7,
    "half": False,
    "device": "cuda:0",
    "agnostic_nms": False,
    "classes": [2, 3, 5, 7],
    "stream": False
})