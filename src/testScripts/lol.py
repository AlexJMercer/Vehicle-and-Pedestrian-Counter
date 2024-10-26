import numpy as np
import cv2

def returnCameraIndexes():
    idx = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(idx)
        if cap.read()[0]:
            arr.append(idx)
            cap.release()
        idx += 1
        i -= 1
    return arr


# show all possible cameras available
# print("Within index range: " + returnCameraIndexes())

# Create numpy array from text file
# classNames = np.genfromtxt("../info/detectClass.txt", dtype=str, delimiter="\n")
# print(classNames)


# classnames = ['bicycle','car','motorbike','bus','truck','person','train','traffic light','vehicle']
# print(classnames)


# import torch

# print(torch.cuda.is_available())

# import cv2
# import numpy as np

# img = np.zeros((512, 512, 3), np.uint8)
# cv2.imshow('Test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import time

# Display current time in hours, minutes, seconds
# print(time.strftime("%H:%M:%S", time.localtime()))



polygons = [
    np.array([
        [0, 0],
        [1080 - 5, 0],
        [1080 - 5, 1300 - 5],
        [0, 1300 - 5]
    ], np.int32),
    np.array([
        [1080 + 5, 0],
        [2160, 0],
        [2160, 1300 - 5],
        [1080 + 5, 1300 - 5]
    ], np.int32),
    np.array([
        [0, 1300 + 5],
        [1080 - 5, 1300 + 5],
        [1080 - 5, 3840],
        [0, 3840]
    ], np.int32),
    np.array([
        [1080 + 5, 1300 + 5],
        [2160, 1300 + 5],
        [2160, 3840],
        [1080 + 5, 3840]
    ], np.int32)
]

print(polygons)