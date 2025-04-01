from box import Box

yolo_vehicle_detection_user_config = Box({

})

yolo_vehicle_detection_model_config = Box({
    "model": "resources/models/yolo/yolo12m.pt",
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



color_detection_user_config = Box({
    "iscc_nbs_colour_system_path": "resources/files/iscc-nbs-colour-system.xlsx",
})