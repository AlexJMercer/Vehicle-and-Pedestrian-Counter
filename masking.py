import cv2
import numpy as np


def get_coordinates(video_capture):
    """Gets a list of 4 coordinate values from the video capture screen.

    Args:
        video_capture: OpenCV video capture object.

    Returns:
        A list of four tuples containing (x, y) coordinates, or None if the user exits before selecting 4 points.
    """

    coordinates = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            print('Coordinates: ', coordinates)

    cv2.namedWindow('Select 4 points')
    cv2.setMouseCallback('Select 4 points', mouse_callback)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if len(coordinates) == 4:
            break

        # Display selected points (optional)
        for point in coordinates:
            cv2.circle(frame, point, 3, (5, 250, 50), -1)

        cv2.imshow('Select 4 points', frame)

        if cv2.waitKey(80) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(coordinates) != 4:
        return None

    return coordinates



def create_mask(video_capture):
    """Creates a mask based on the coordinates provided.

    Args:
        None

    Returns:
        A mask image.
    """
    if ( input("Use existing mask? (y/n): ") == 'y' ):
        try:
            mask = cv2.imread('./info/mask.png')
            return mask
        except FileNotFoundError:
            print('Error: Mask file not found')
            print('Creating new mask')
    else:
        pass

    if not video_capture.isOpened():
        print('Error: Video file not found')
        return None

    ret, frame = video_capture.read()

    if not ret:
        print('Error: Frame not found')
        return None

    video_capture.release()

    coordinates = np.loadtxt('./info/maskCoords.txt', delimiter=',', dtype=int)

    points = np.array(coordinates, np.int32)

    # Create the Mask
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [points], (255, 255, 255))

    # Change the mask to 3 channels
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Displaying Mask
    cv2.imshow('Mask', mask)

    cv2.imwrite('./info/mask.png', mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return mask



def set_counter_line_coordinates(video_capture):
    """Sets the coordinates for the counter line.

    Args:
        video_capture: OpenCV video capture object.

    Returns:
        A list of two tuples containing (x, y) coordinates.
    """

    coordinates = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            print('Coordinates: ', coordinates)

    cv2.namedWindow('Select 2 points')
    cv2.setMouseCallback('Select 2 points', mouse_callback)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if len(coordinates) == 2:
            break

        # Display selected points (optional)
        for point in coordinates:
            cv2.circle(frame, point, 3, (5, 250, 50), -1)

        cv2.imshow('Select 2 points', frame)

        if cv2.waitKey(80) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(coordinates) != 2:
        return None

    return coordinates