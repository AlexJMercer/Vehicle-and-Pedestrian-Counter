import math
import time

import cv2
import numpy as np

import supervision as sv

# from sort import *


def startDetection(videoCap, yoloModel, limitsCoords, detectionMask):
    '''
    Function to start the detection and tracking of vehicles
    in the video stream.
    '''

    # This is supposed to contain the IDs (Number Plates) of the vehicles to be tracked
    countList = []
    totalCount = 1

    # 2D for recording the crossing of a vehicle with time-stamp
    vehicleCrossings = []

    
    # Load the class names
    classNames = np.genfromtxt("./info/detectClass.txt", dtype=str, delimiter="\n").tolist()

    # Vehicles get counted when they pass this line : (x1, y1) to (x2, y2)
    limits = np.array(limitsCoords).flatten().tolist()
    
    

    # Inititalize ByteTracker from Supervision
    # tracker = sv.ByteTrack()

    tracker = sv.ByteTrack(
        track_activation_threshold=0.2,
        lost_track_buffer=30,
        # minimum_matching_threshold=0.6,
        minimum_consecutive_frames=10,
        frame_rate=videoCap.get(cv2.CAP_PROP_FPS)
    )

    # Initialize the BoundingBox and Label Annotators with Properties
    boundingAnnotator = sv.BoxCornerAnnotator(
        color=sv.ColorPalette.from_hex(["#00FF00"]),
        thickness=1,
    )

    labelAnnotator = sv.LabelAnnotator(
        text_padding=4,
    )


    # Line Counter and Line Annotator
    lineCounter = sv.LineZone(
        start = sv.Point(limits[0], limits[1]),
        end   = sv.Point(limits[2], limits[3])
    )

    lineAnnotator = sv.LineZoneAnnotator(
        text_thickness=2,
        text_scale=0.9,
        thickness=2
    )


    # Main Loop to process frames
    while True:
        cap, frame = videoCap.read()

        if not cap:
            break

        # Get the region of interest
        imgRegion = cv2.bitwise_and(frame, detectionMask)

        # Get the results from the YOLO model
        results = yoloModel(imgRegion, stream=True, agnostic_nms=True)
        
        # Loop through the results
        for result in results:

            # Initialize the detection array
            detection = sv.Detections.from_ultralytics(result)
            detection = tracker.update_with_detections(detection)

            # Update line counter
            crossed_in, crossed_out = lineCounter.trigger(detection)

            labels = []
            for _, _, confidence, class_id, tracker_id, _ in detection:
                if confidence is not None and yoloModel.model.names[class_id] in classNames:
                    label = f"ID #{tracker_id} {yoloModel.model.names[class_id]} {math.floor(confidence*100)}%"

                    if tracker_id not in countList:
                        countList.append(totalCount)
                        totalCount += 1
                    
                    if (np.any(crossed_in) or np.any(crossed_out)) and tracker_id in countList and tracker_id not in [x[0] for x in vehicleCrossings]:
                        vehicleCrossings.append((
                            tracker_id, 
                            time.strftime("%H:%M:%S", time.localtime())
                        ))

                else:
                    label = f"ID #{tracker_id} {yoloModel.model.names[class_id]} Unknown Confidence"
                
                labels.append(label)
                
           


            # Attributes in Detection array (for debugging)
            # first_detection = detection[0]
            # first_10_detections = detection[0:10]
            # class_0_detections = detection[detection.class_id == 0]
            # high_confidence_detections = detection[detection.confidence > 0.5]
            
            # print(first_detection)
            # print(first_10_detections)
            # print(class_0_detections)
            # print(high_confidence_detections)



            # Annotate the frame with the bounding boxes
            annotated_frame = boundingAnnotator.annotate(
                scene=frame.copy(), 
                detections=detection
            )

            # Annotate the frame with the labels
            annotated_frame = labelAnnotator.annotate(
                scene=annotated_frame, 
                detections=detection, 
                labels=labels
            )

            # Annotate the frame with the line counter
            annotated_frame = lineAnnotator.annotate(
                frame=annotated_frame, 
                line_counter=lineCounter
            )

            # Display the current count of vehicles
            cv2.putText(
                img=annotated_frame, 
                text=f'Vehicles Detected: {len(countList)}', 
                org=(50, 50), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.9, 
                color=(255, 255, 255), 
                thickness=2
            )

            # Display the line for counting vehicles
            cv2.line(
                img=annotated_frame, 
                pt1=(limits[0], limits[1]), 
                pt2=(limits[2], limits[3]), 
                color=(255, 255, 255), 
                thickness=2
            )

            # Display the frame
            cv2.imshow('Output View', annotated_frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return countList, vehicleCrossings


    # while True:
    #     sucess, img = videoCap.read()
    #     imgRegion = cv2.bitwise_and(img, detectionMask)
    #     results = yoloModel(imgRegion, stream=True)
    #     detection = np.empty((0, 5))

    #     for r in results:
    #         boxes = r.boxes
    #         for box in boxes:
    #             # Bounding Box

    #             x1, y1, x2, y2 = box.xyxy[0]
    #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #             print(x1, y1, x2, y2)
    #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #             # Confidence value
    #             conf = math.ceil((box.conf[0]*100))/100            
                
    #             # Class name
    #             classIndex = int(box.cls[0])
    #             if classIndex < len(classNames):
    #                 current_class = classNames[classIndex]
    #             else:
    #                 pass

    #             if current_class == 'car' and conf > 0.5:
    #                 current_array = np.array([[x1, y1, x2, y2, conf]])
    #                 detection = np.vstack((detection, current_array))

    #     # Update tracker
    #     resultsTracker = tracker.update(detection)
    #     for result in resultsTracker:
    #         x1, y1, x2, y2, id = result
    #         x1, y1, x2, y2, id  = int(x1), int(y1), int(x2), int(y2), int(id)
    #         w, h = x2 - x1, y2 - y1

    #         cv2.putText(img, f'{classNames[id]}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    #         cv2.circle(img, (cx, cy), 3, (0, 255, 255), cv2.FILLED)

    #         if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15:
    #             if id not in totalCount:
    #                 totalCount.append(id)

    #             cv2.line(img, (cx, cy), (cx, cy), (255, 0, 0), 1)
    #             cv2.putText(img, f' Count : {len(totalCount)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #     cv2.imshow('Image', img)
    #     cv2.waitKey(27)
