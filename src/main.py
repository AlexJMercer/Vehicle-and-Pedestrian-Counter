from ultralytics import YOLO 
import numpy as np
import cv2

from env_var import *

from masking import *
from monitorTraffic import startDetection
from plotGraph import *




'''
Main function of the program

'''

if __name__ == '__main__':
    
# Check video file
    videoCapture = cv2.VideoCapture(VIDEO_PATH)             # Example Video file
    # videoCapture = cv2.VideoCapture(0)                        # Source: Webcam

    
    if not videoCapture.isOpened():
        print('Error: Video file not found')
        exit()

# Load YOLOv8 model
    yoloModel = YOLO('yolov8n.pt')

    if not yoloModel.model:
        print('Error: Model not found')
        exit()


    coordinates = []
    limitsCoords = []

# Get coordinates for masking
#
# If coordinates are already saved in file, load them instead
# unless user chooses to override

    if (input('Load coordinates from file? (y/n): ') == 'y'):
        try:
            coordinates = np.load('./info/maskCoords.npy', allow_pickle=True)
            
            if not coordinates.any():
                print('Error: No coordinates found in file')
                print('Gathering new coordinates from Video Capture')
                coordinates = get_coordinates(videoCapture)
        
        except FileNotFoundError:
            print('Error: File not found')
            print('Gathering new coordinates from Video Capture')
            coordinates = get_coordinates(videoCapture)
        
    else:
        coordinates = get_coordinates(videoCapture)

    # if coordinates is not None:
    #     print('Error: No coordinates selected')
    #     exit()
    
    # Save coordinates to file
    print('Coordinates: ', coordinates)
    
    np.save('./info/maskCoords.npy', np.array(coordinates))


# Select and get coordinates for drawing a line on the screen
    # if (input('Load line coordinates from file? (y/n): ') == 'y'):
    #     try:
    #         limitsCoords = np.loadtxt('./info/lineCoords.txt', delimiter=',', dtype=int)
            
    #         if not limitsCoords.any():
    #             print('Error: No coordinates found in file')
    #             print('Gathering new coordinates from Video Capture')
    #             limitsCoords = set_counter_line_coordinates(videoCapture)
        
    #     except FileNotFoundError:
    #         print('Error: File not found')
    #         print('Gathering new coordinates from Video Capture')
    #         limitsCoords = set_counter_line_coordinates(videoCapture)
        
    # else:
    #     limitsCoords = set_counter_line_coordinates(videoCapture)

    # if limitsCoords is None or len(limitsCoords) != 2:
    #     print('Error: No coordinates selected')
    #     exit()
    
    # # Save coordinates to file
    # limitsCoords = np.flip(limitsCoords, axis=0)
    # np.savetxt('./info/lineCoords.txt', limitsCoords, delimiter=',', fmt='%d')
    
    # print('Line Coordinates: ', limitsCoords)


# Now we create the mask using the coordinates obtained
    # detectionMask = create_mask(videoCapture)
    
    # if detectionMask is None or not detectionMask.any():
    #     print('Error: Mask not created')
    #     exit()

    # print('Mask created')


# Launch the main detection loop
    # countList, vehicleCrossings, pedestrianCount = startDetection(videoCapture, yoloModel, coordinates)
    startDetection(videoCapture, yoloModel, coordinates)

#     print(f'\n\nTotal Vehicles Detected: { len(countList) }')
#     print("List of all Vehicle IDs: ", countList)

#     print("\n\nNumber of Pedestrians detected :\n", pedestrianCount)

#     print("\n\nVehicles that passed with time stamps:\n", vehicleCrossings)


# # Display a graphical representation of number of vehicles vs time
#     if vehicleCrossings:
#         plot_graph_vehicle_count(vehicleCrossings)
