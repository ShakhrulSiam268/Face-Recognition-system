import warnings
warnings.filterwarnings("ignore")
import cv2
import os 
import numpy as np
import time
import pickle
import torch
import pandas as pd
from datetime import datetime
from Retinaface_gpu.Retina_gpu import RetinaGPU
from resnet100.arcface_recognition_r100 import ArcfaceR100
from utils.face_utils import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def face_recognition(image, scale):
    faces, face_region = retina.extract_faces(image, scale, alignment=True) # Retina-GPU
    face_region = np.array(face_region)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    org_img = image.copy()
    detected_names = []
    print('Total Face Found : ', len(faces))

    for face in faces:
        try:
            face = cv2.resize(face, (112, 112))
            embd = np.array(arcface_model.get_feature(face))
            probabilities = np.dot(known_embeddings,embd.T)
            index = np.argmax(probabilities)
            score = np.max(probabilities)

            if score >= threshold:
                name_id = int(known_labels[index])
                name = names[name_id]
            else:
                name = 'unknown'
            detected_names.append(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            if name != 'unknown':
                print('    Name :    {}     Score :    {}    Time: {}'.format(name, round(score, 3), current_time))
        except:
            print('Face Loading error, shape:', face.shape)

    detected_names = np.array(detected_names)
    resize_to = 1280
    img_resized = cv2.resize(org_img,(resize_to,resize_to))
    face_region_resized = []
    for region in face_region:
        
        resized_region = [region[0]*(resize_to/org_img.shape[1]), region[1]*(resize_to/org_img.shape[0]), region[2]*(resize_to/org_img.shape[1]), region[3]*(resize_to/org_img.shape[0])]
        face_region_resized.append(resized_region) 
    face_region_resized = np.array(face_region_resized)
    for i in range(len(face_region)):
        image = draw_boxes(img_resized, face_region_resized[i], detected_names[i], 1)
    return image


weight_path = './models/r100_Glint360kCosface.pth'
arcface_model = ArcfaceR100()
model = arcface_model.load_insightface_pytorch_model(weight_path)
threshold = 0.5
known_embeddings = pd.read_csv("./embed_data/embed_data_r100.csv",header=None)
known_labels = pd.read_csv("./embed_data/true_label_r100.csv",header=None)
known_labels = np.array(known_labels)
with open('./embed_data/names_r100.pkl', 'rb') as handle:
    names = pickle.load(handle)


if torch.cuda.is_available():
    retina = RetinaGPU(0)   # (-1) for CPU, (0) for GPU
    print('Cuda Available, Running on GPU...')
    print('Device Name : ',torch.cuda.get_device_name(0))
else:
    retina = RetinaGPU(-1)
    print('Cuda Not Found, Running on CPU...')

scale = 0.6

t1 = time.time()
file_name = '8.jpg'
frame = cv2.imread('./test/'+file_name)
image = face_recognition(frame, scale)
image = cv2.resize(image, (640, 640))
t2 = time.time()
print('Processing Time : {} s'.format(round((t2-t1),3)))
cv2.imwrite('./test/results/'+file_name, image)

cv2.imshow('Face Recognition MIS', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

