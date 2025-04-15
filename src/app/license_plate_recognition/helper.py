import supervision as sv


def is_license_plate_inside_vehicle(
    vehicle_detection: sv.Detections,
    license_plate_detection: sv.Detections,
) -> list[bool]:
    # Get the bounding boxes of the vehicle and license plate
    vehicle_bboxs = vehicle_detection.xyxy
    license_plate_bboxs = license_plate_detection.xyxy

    inside = [False] * len(license_plate_bboxs) 
    for vehicle_bbox in vehicle_bboxs:
        for index, license_plate_bbox in enumerate(license_plate_bboxs):
            # Check if the license plate is within the vehicle bounding box
            if (
                license_plate_bbox[0] >= vehicle_bbox[0]
                and license_plate_bbox[1] >= vehicle_bbox[1]
                and license_plate_bbox[2] <= vehicle_bbox[2]
                and license_plate_bbox[3] <= vehicle_bbox[3]
            ):
                inside[index] = True
    return inside   