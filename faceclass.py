import warnings
warnings.filterwarnings("ignore")
import cv2
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from Retinaface_gpu.Retina_gpu import RetinaGPU
from Arcface_custom import ArcFaceRecog
from utils.face_utils import *
import time


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FaceClass:
    def __init__(self, threshold=0.7):
        arcface_model_path = './models/model.tflite'
        self.face_rec = ArcFaceRecog.ArcFace(arcface_model_path)
        self.threshold = threshold
        face_db = pd.read_csv('./embed_data/MIS_Face_DB.csv')
        self.known_embeddings = face_db.iloc[:, 2:]
        known_labels = face_db.iloc[:, 0]
        self.known_labels = np.array(known_labels)
        self.names = face_db.iloc[:, 1].unique()

        if torch.cuda.is_available():
            self.retina = RetinaGPU(0)  # (-1) for CPU, (0) for GPU
            print('Cuda Available, Running on GPU...')
            print('Device Name : ', torch.cuda.get_device_name(0))
        else:
            self.retina = RetinaGPU(-1)
            print('Cuda Not Found, Running on CPU...')

    def face_recognition(self, image, scale):
        t1 = time.time()
        faces, face_region = self.retina.extract_faces(image, scale, alignment=True)
        face_region = np.array(face_region)
        org_img = image.copy()
        detected_names = []
        for face in faces:
            try:
                face = cv2.resize(face, (112, 112))
                embd = self.face_rec.calc_emb(face)
                embd = np.array(embd)
                probabilities = np.dot(self.known_embeddings, embd.T)
                index = np.argmax(probabilities)
                score = np.max(probabilities)

                if score >= self.threshold:
                    name_id = int(self.known_labels[index])
                    name = self.names[name_id]
                else:
                    name = 'unknown'
                detected_names.append(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                # if name != 'unknown':
                #     print('    Name :    {}     Score :    {}    Time: {}'.format(name, round(score, 3), current_time))
            except:
                print('Face Loading error, shape:', face.shape)

        detected_names = np.array(detected_names)
        resize_to = 1280
        img_resized = cv2.resize(org_img, (resize_to, resize_to))
        image_out = img_resized.copy()
        face_region_resized = []
        for region in face_region:
            resized_region = [region[0]*(resize_to/org_img.shape[1]), region[1]*(resize_to/org_img.shape[0]), region[2]*(resize_to/org_img.shape[1]), region[3]*(resize_to/org_img.shape[0])]
            face_region_resized.append(resized_region)
        face_region_resized = np.array(face_region_resized)

        for i in range(len(face_region)):
            image_out = draw_boxes(img_resized, face_region_resized[i], detected_names[i], scale)

        t2 = time.time()
        fps = round(1 / (t2 - t1), 1)
        cv2.rectangle(image_out, (40, 0), (300, 70), (0, 0, 0), -1)
        cv2.putText(image_out, f'FPS : {fps}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 128, 0), thickness=2, lineType=1)
        return image_out, detected_names

