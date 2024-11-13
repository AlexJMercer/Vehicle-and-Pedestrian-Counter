import math
import time

import cv2
import numpy as np

import supervision as sv
import os

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


def assign_zone_labels(zones, loadLabels):
    zone_labels = {}

    if loadLabels:
        # Check if the file exists
        if os.path.exists('./info/zoneLabels.txt'):
            # Load data with np.loadtxt if file exists
            try:
                loaded_labels = np.loadtxt('./info/zoneLabels.txt', dtype=str, delimiter=',')
                
                # Ensure loaded_labels is not empty
                if loaded_labels.size == 0:
                    raise ValueError("Empty label file.")

                # Map zones to labels from the file
                for i, zone in enumerate(zones):
                    zone_labels[zone] = loaded_labels[i] if i < len(loaded_labels) else f"Zone_{i}"
            
            except Exception as e:
                print(f'Error loading labels from file: {e}')
                print('Gathering new zone labels')
                
                # Prompt for input if there was an issue loading from file
                for i, zone in enumerate(zones):
                    label = input(f"Enter label for zone {i}: ")
                    zone_labels[zone] = label
        
        else:
            print('File not found or empty. Gathering new zone labels.')
            
            # Prompt for input if the file doesn’t exist
            for i, zone in enumerate(zones):
                label = input(f"Enter label for zone {i}: ")
                zone_labels[zone] = label
    else:
        # Prompt for input if loadLabels is False
        for i, zone in enumerate(zones):
            label = input(f"Enter label for zone {i}: ")
            zone_labels[zone] = label

    return zone_labels



def startDetection(videoCap, model, polygons, loadLabels):

    colors = sv.ColorPalette.DEFAULT

    # 0 - person
    # 1 - bicycle
    # 2 - car
    # 3 - motorcycle
    # 5 - bus
    # 7 - truck
    selected_classes = [0, 1, 2, 3, 5, 7]


    zones = [
        sv.PolygonZone(
            polygon=polygon,
        )
        for polygon
        in polygons
    ]


    zone_labels = assign_zone_labels(zones, loadLabels)


    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
        )
        for index, zone
        in enumerate(zones)
    ]

    box_annotators = [
        sv.BoxCornerAnnotator(
            color=colors.by_idx(index),
        )
        for index
        in range(len(polygons))
    ]

    label_annotator = sv.LabelAnnotator(
        text_padding=2,
        text_position=sv.Position.TOP_LEFT
    )

    while True:
        ret, frame = videoCap.read()

        if not ret:
            break

        results = model(
            frame,
            imgsz=640,
            verbose=False
        )

        for result in results:
            detections = sv.Detections.from_ultralytics(result)

            for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                mask = zone.trigger(detections)
                detections_filtered = detections[mask & (detections.confidence > 0.5) & np.isin(detections.class_id, selected_classes)]

                labels = [
                    f"{model.model.names[class_id]} {math.floor(conf * 100)}%"
                    for _, _, conf, class_id, _, _ in detections_filtered
                ]

                frame = zone_annotator.annotate(
                    scene=frame,
                )

                frame = box_annotator.annotate(
                    scene=frame,
                    detections=detections_filtered
                )

                frame = label_annotator.annotate(
                    scene=frame,
                    detections=detections_filtered,
                    labels=labels
                )


                zone_text = f"{zone_labels[zone]} zone"

                # Place label on top of the zone in the video frame
                top_left = zone.polygon[0]  # Get the top-left corner of the zone box
                cv2.putText(
                    frame,
                    zone_text,
                    (int(top_left[0]), int(top_left[1]) - 10),  # Adjust position above the top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,                                        # Font scale
                    (255, 255, 255),                            # White color for text
                    2,                                          # Thickness
                    cv2.LINE_AA
                )

                # Print the name of the zone and the number of detections in console
                print(f"{zone_labels[zone]} Zone: {zone.current_count} detections")

            print()

        # Compare the number of pedestrians and vehicles in each zone every 200 frames
        if videoCap.get(cv2.CAP_PROP_POS_FRAMES) % 200 == 0:
            evaluate_traffic_conditions(zones, zone_labels)


        cv2.imshow('Output View', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Use waitKey(1) for real-time video / camera feed
            break

    videoCap.release()
    cv2.destroyAllWindows()



def evaluate_traffic_conditions(zones, zone_labels):
        # If total of zone 1 and zone 2 is less than total of zone 3, print a message
        
        # If the number of vehicles on road is very less, and pedestrians are more,
        # let the pedestrian pass first
        
        # Calculate total number of detections in zones whose labels are not 'Pedestrian'
        vehicle_count = sum(
                    zone.current_count
                    for zone in zones
                    if zone_labels[zone] != "Pedestrian"
                )

        pedestrian_count = sum(
                    zone.current_count
                    for zone in zones
                    if zone_labels[zone] == "Pedestrian"
                )
        
        # Since pedestrians will always be more, let's add a offset value to the vehicle count
        if (vehicle_count + 5) < pedestrian_count:
            # Let pedestrians pass first and stop vehicles
            print("Pedestrians have right of way. Let pedestrians pass...")
            # Replace this line with a function to control the traffic signal
            print("Traffic Signal: RED")
            time.sleep(5)
        
        else:
            # Let vehicles pass and stop pedestrians
            print("Vehicles have right of way. Proceed with caution...")
            # Replace this line with a function to control the traffic signal
            print("Traffic Signal: GREEN")
            