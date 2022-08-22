import cv2
import time
from facerec import FaceRecognitionSystem

scale = 0.6
t1 = time.time()
file_name = '8.jpg'
frame = cv2.imread('./test/' + file_name)

FRC = FaceRecognitionSystem()
image = FRC.face_recognition(frame, scale)
image = cv2.resize(image, (640, 640))
t2 = time.time()
print('Processing Time : {} s'.format(round((t2 - t1), 3)))

cv2.imwrite('./test/results/' + file_name, image)
cv2.imshow('Face Recognition MIS', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
