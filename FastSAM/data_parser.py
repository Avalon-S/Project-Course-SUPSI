import struct
import numpy as np
import cv2

def parse_received_data(data):
    """
    Parse the data received from HoloLens 2.

    Data format:
    - First 4 bytes: image data length (int)
    - Next 4 bytes: coordinate data length (int)
    - Then, the image data follows
    - Followed by the coordinate data (UTF-8 encoded string)

    Returns:
    - image: OpenCV image object
    - fingertip_position: (x, y) float tuple
    """
    # Read image data length
    image_length = struct.unpack('I', data[0:4])[0]
    # Read coordinate data length
    position_length = struct.unpack('I', data[4:8])[0]

    # Read image data
    image_data = data[8:8+image_length]
    # Read coordinate data
    position_data = data[8+image_length:8+image_length+position_length]

    # Decode the image
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Decode the coordinate data
    position_string = position_data.decode('utf-8')
    x_str, y_str = position_string.split(',')[:2]  # Only extract x and y
    fingertip_position = (float(x_str), float(y_str))

    return image, fingertip_position
