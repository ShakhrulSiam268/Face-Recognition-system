import numpy as np
import cv2
import sys
import traceback

def exception_printer(error_message):

    exception_message = traceback.format_exc(limit=2)

    print('\n' + str(error_message))
    print(exception_message)

class ImageLoader:
    def __init__(self):
        print('ImageLoader Loaded...')
        return

    @staticmethod
    def loader(image_path, convert_color=None, transpose=(0, 1, 2), dtype=np.float32):
        try:

            image = image_path.copy()

            if convert_color is not None:
                image = cv2.cvtColor(image, convert_color)
            image = np.transpose(image, transpose)
            image = np.array(image, dtype=dtype)
            image = np.array([image])

            return image

        except Exception as ex:
            exception_printer('Load image \'' + str(image_path) + '\' failed.')