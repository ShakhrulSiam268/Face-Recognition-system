import cv2
import os
import argparse
from utils.videocapture import CustomVideoCapture
from faceclass import FaceClass

scale = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--im", default=False, action="store_true")
    parser.add_argument('--path', type=str, default='./test/8.jpg', help='input image path')
    args = parser.parse_args()

    if args.im:
        image_path = args.path
        process_image(image_path)
    else:
        process_video()


def process_image(image_path):
    if os.path.isfile(image_path):
        frame = cv2.imread(image_path)
    else:
        print('File Not Exists')
        exit()
    FC = FaceClass(threshold=0.65)
    image, detected_names = FC.face_recognition(frame, 1)
    image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)
    cv2.imshow('Face Recognition MIS', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video():
    url = 0
    cap = CustomVideoCapture(url)
    FC = FaceClass(threshold=0.7)
    while True:
        frame = cap.read()
        image, detected_names = FC.face_recognition(frame, scale)
        image = cv2.resize(image, (1500, 900))
        cv2.imshow('Face Recognition MIS', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
