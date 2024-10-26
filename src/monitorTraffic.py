from env_var import *

import math
import time

import cv2
import numpy as np

import supervision as sv

# from sort import *


# def startDetection(videoCap, yoloModel, limitsCoords):
#     '''
#     Function to start the detection and tracking of vehicles
#     in the video stream.
#     '''

#     # This is supposed to contain the IDs (Number Plates) of the vehicles to be tracked
#     countList = []

#     # List to contains the IDs of the pedestrians
#     pedestrianList = []

#     # Add a pedestrian count to the list
#     pedestrianCount = 1
#     totalCount = 1

#     # 2D for recording the crossing of a vehicle with time-stamp
#     vehicleCrossings = []

    
#     # Load the class names
#     classNames = np.genfromtxt("./info/detectClass.txt", dtype=str, delimiter="\n").tolist()

#     # Vehicles get counted when they pass this line : (x1, y1) to (x2, y2)
#     # limits = np.array(limitsCoords).flatten().tolist()


#     # Inititalize ByteTracker from Supervision

#     tracker = sv.ByteTrack(
#         track_activation_threshold=0.2,
#         lost_track_buffer=30,
#         # minimum_matching_threshold=0.6,
#         minimum_consecutive_frames=10,
#         frame_rate=videoCap.get(cv2.CAP_PROP_FPS)
#     )

#     video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)
    
#     # Get optimal line thickness depending upon the resolution of the video
#     thickness = sv.calculate_optimal_line_thickness(
#         resolution_wh = video_info.resolution_wh
#     )
    
#     # Get optimal text scale from video
#     text_scale = sv.calculate_optimal_text_scale(
#         resolution_wh = video_info.resolution_wh
#     )


        
#     # Make the zones
#     zones = [
#         sv.PolygonZone(
#             polygon=polygon,
#             frame_resolution_wh=video_info.resolution_wh
#         ) 
#         for polygon 
#         in limitsCoords
#     ]

#     zone_annotators = [
#         sv.PolygonZoneAnnotator(
#             zone=zone,
#             color=sv.Color.from_hex("#00FF00"),
#             thickness=thickness,
#             text_thickness=thickness,
#             text_scale=text_scale
#         )
#         for index, zone 
#         in enumerate(zones)
#     ]

#     # Initialize the BoundingBox, Label & Trace Annotators with Properties
#     boundingAnnotator = [
#         sv.BoxCornerAnnotator(
#             color=sv.ColorPalette.from_hex(["#00FF00"]),
#             thickness=thickness,
#         )
#         for _ 
#         in range(len(zones)) 
#     ]
    

#     labelAnnotator = [
#             sv.LabelAnnotator(
#             text_padding=4,
#         )
#     ]

#     # Line Counter and Line Annotator
#     # lineCounter = sv.LineZone(
#     #     start = sv.Point(limits[0], limits[1]),
#     #     end   = sv.Point(limits[2], limits[3])
#     # )

#     # lineAnnotator = sv.LineZoneAnnotator(
#     #     text_thickness=thickness,           # Default 2
#     #     text_scale=text_scale,              # Default 0.9
#     #     thickness=thickness                 # Default 2
#     # )


#     # Main Loop to process frames
#     while True:
#         cap, frame = videoCap.read()

#         if not cap:
#             break

#         # Get the region of interest
#         # imgRegion = cv2.bitwise_and(frame, detectionMask)

#         # Get the results from the YOLO model
#         results = yoloModel(
#             frame, 
#             stream=True, 
#             agnostic_nms=True,
#             verbose=False
#         )
        
#         # Loop through the results
#         for result in results:

#             # Initialize the detection array
#             detection = sv.Detections.from_ultralytics(result)
#             detection = tracker.update_with_detections(detection)

#             labels = []
#             for _, _, confidence, class_id, tracker_id, _ in detection:

#                 if confidence is not None and yoloModel.model.names[class_id] in classNames:
                    
#                     label = f"ID #{tracker_id} {yoloModel.model.names[class_id]} {math.floor(confidence*100)}%"
                    
#                     # Detect and count pedestrians if any
#                     if yoloModel.model.names[class_id] == 'person' and tracker_id not in pedestrianList:
#                         label = f"Pedestrian {pedestrianCount} {math.floor(confidence*100)}%"
#                         pedestrianList.append(tracker_id)
#                         pedestrianCount += 1

#                     elif tracker_id not in countList:
#                         countList.append(totalCount)
#                         totalCount += 1

#                 else:
#                     label = f"ID #{tracker_id} {yoloModel.model.names[class_id]} Unknown Confidence"
                
#                 labels.append(label)
                
           


#             # Attributes in Detection array (for debugging)
#             # first_detection = detection[0]
#             # first_10_detections = detection[0:10]
#             # class_0_detections = detection[detection.class_id == 0]
#             # high_confidence_detections = detection[detection.confidence > 0.5]
            
#             # print(first_detection)
#             # print(first_10_detections)
#             # print(class_0_detections)
#             # print(high_confidence_detections)



#             # Annotate the frame with the bounding boxes
#             # annotated_frame = boundingAnnotator.annotate(
#             #     scene=frame.copy(), 
#             #     detections=detection
#             # )

#             # # Annotate the frame with the labels
#             # annotated_frame = labelAnnotator.annotate(
#             #     scene=annotated_frame, 
#             #     detections=detection, 
#             #     labels=labels
#             # )

#             for zone, zone_annotator, bounding, label in zip(zones, zone_annotators, boundingAnnotator, labelAnnotator):
#                 mask = zone.trigger(detection)
#                 detection_filtered = detection[mask]
                
#                 frame = zone_annotator.annotate(
#                     scene=frame, 
#                 )
                
#                 if isinstance(detection_filtered, sv.Detections):
                    
#                     frame = bounding.annotate(
#                         scene=frame, 
#                         detections=detection_filtered
#                     )
                
#                     frame = label.annotate(
#                         scene=frame, 
#                         detections=detection_filtered, 
#                         labels=labels
#                     )

#             # Display the current count of vehicles
#             cv2.putText(
#                 img=frame, 
#                 text=f'Vehicles Detected: {len(countList)}', 
#                 org=(50, 50), 
#                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
#                 fontScale=text_scale,               # Originally 
#                 color=(255, 255, 255), 
#                 thickness=2
#             )


#             # Display the frame
#             cv2.imshow('Output View', frame)
        

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     videoCap.release()
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)
    
#     return countList, vehicleCrossings, pedestrianCount

    # ==========================================================================
    



def startDetection(videoCap, model, polygons):

    video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)

    colors = sv.ColorPalette.DEFAULT

    # Get optimal line thickness depending upon the resolution of the video
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh = video_info.resolution_wh
    )
    
    # Get optimal text scale from video
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh = video_info.resolution_wh
    )

    zones = [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=video_info.resolution_wh
        )
        for polygon
        in polygons
    ]

    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=thickness,
            text_thickness=thickness,
            text_scale=text_scale
        )
        for index, zone
        in enumerate(zones)
    ]


    box_annotators = [
        sv.BoxCornerAnnotator(
            color=colors.by_idx(index),
            thickness=thickness,
            )
        for index
        in range(len(polygons))
    ]

    while True:
        ret, frame = videoCap.read()

        if not ret:
            break

        results = model(
            frame,
            imgsz=1280,
            verbose=True
        )

        for result in results:
            detections = sv.Detections.from_ultralytics(result)

            for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                mask = zone.trigger(detections)
                detections_filtered = detections[mask]

                frame = zone_annotator.annotate(
                    scene=frame,
                )

                frame = box_annotator.annotate(
                    scene=frame,
                    detections=detections_filtered
                )

        cv2.imshow('Output View', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    videoCap.release()
    cv2.destroyAllWindows()