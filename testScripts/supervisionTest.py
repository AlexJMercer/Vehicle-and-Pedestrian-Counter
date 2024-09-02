from typing import Tuple
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def main():
    cap = cv2.VideoCapture(0)
    webcam_resolution: Tuple[int, int] = (640, 480)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2
    )

    zone_polygon = (ZONE_POLYGON * np.array(webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(640, 480))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone = zone, 
        color = sv.Color.from_hex("#00FF00"),
        thickness = 2,
        text_thickness = 4,
        text_scale = 2
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov5(result
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)      
        
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()