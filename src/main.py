from ultralytics import YOLO 
import numpy as np
import cv2

from masking import *
from monitorTraffic import *
from plotGraph import *

from buildGUI import *


'''
Main function of the program

'''

# if __name__ == '__main__':
    
# # Check video file
#     videoCapture = cv2.VideoCapture(VIDEO_PATH)             # Example Video file
#     # videoCapture = cv2.VideoCapture(0)                        # Source: Webcam

    
#     if not videoCapture.isOpened():
#         print('Error: Video file not found')
#         exit()

# # Load YOLOv8 model
#     yoloModel = YOLO('yolov8n.pt')

#     if not yoloModel.model:
#         print('Error: Model not found')
#         exit()


#     coordinates = []
#     limitsCoords = []

# Get coordinates for masking
#
# If coordinates are already saved in file, load them instead
# unless user chooses to override

    # if (input('Load coordinates from file? (y/n): ') == 'y'):
    #     try:
    #         coordinates = np.load('./info/maskCoords.npy', allow_pickle=True)
            
    #         if not coordinates.any():
    #             print('Error: No coordinates found in file')
    #             print('Gathering new coordinates from Video Capture')
    #             coordinates = get_coordinates(videoCapture)
        
    #     except FileNotFoundError:
    #         print('Error: File not found')
    #         print('Gathering new coordinates from Video Capture')
    #         coordinates = get_coordinates(videoCapture)
        
    # else:
    #     coordinates = get_coordinates(videoCapture)

    # # if coordinates is not None:
    # #     print('Error: No coordinates selected')
    # #     exit()
    
    # # Save coordinates to file
    # print('Coordinates: ', coordinates)
    
    # np.save('./info/maskCoords.npy', np.array(coordinates))


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
    # startDetection(videoCapture, yoloModel, coordinates)

#     print(f'\n\nTotal Vehicles Detected: { len(countList) }')
#     print("List of all Vehicle IDs: ", countList)

#     print("\n\nNumber of Pedestrians detected :\n", pedestrianCount)

#     print("\n\nVehicles that passed with time stamps:\n", vehicleCrossings)


# # Display a graphical representation of number of vehicles vs time
#     if vehicleCrossings:
#         plot_graph_vehicle_count(vehicleCrossings)



if __name__ == '__main__':

    # Create GUI
    root = tk.Tk()
    app = BuildGUI(root)
    root.mainloop()
    
    # Get values from GUI
    video_path, model_path, load_coords, load_labels = app.get_values()
    
    # Print or process the values
    print("Video Path:", video_path)
    print("YOLO Model Path:", model_path)
    print("Load Coordinates:", load_coords)
    print("Load Labels:", load_labels)
    print()

    # Check video file
    if video_path is int:
        videoCapture = cv2.VideoCapture(video_path)                        # Source: Webcam
    else:
        videoCapture = cv2.VideoCapture(str(video_path))             # Example Video file

    if not videoCapture.isOpened():
        print('Error: Video file not found')
        exit()

    
    # Load YOLOv8 model
    yoloModel = YOLO(str(model_path))

    if not yoloModel.model:
        print('Error: Model not found')
        exit()


    # Arrays to store Coordinates
    coordinates = []

    # Check if coordinates are to be loaded from file
    if load_coords:
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
    print('Coordinates:\n', coordinates)
    print()
    
    # Save coordinates to file
    np.save('./info/maskCoords.npy', np.array(coordinates))


    # Start the main detection loop
    startDetection(videoCapture, yoloModel, coordinates, load_labels)