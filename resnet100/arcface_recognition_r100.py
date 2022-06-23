import warnings
warnings.filterwarnings("ignore")
import sys
import os
import argparse
import cv2
import numpy as np
import torch
import time
import pickle

from resnet100.tools.pytorch_model_loader import PyTorchModelLoader
from resnet100.utils.image_loader import ImageLoader
#from resnet100.utils.compare_util import Compare_Util, CompareDistanceType


class ArcfaceR100():
    def __init__(self):
        if torch.cuda.is_available():
            print('Runing on CUDA Device')
        else:
            print('Running on CPU')

        #self.network = 'r100'
        #self.weight_path =  None
        #self.model_path = './models/r100_Glint360kCosface.pth'

    def load_insightface_pytorch_model(self,weight_path):
        self.network = 'r100'
        self.model_path =  None
        self.weight_path = weight_path


        pytorch_model_loader = PyTorchModelLoader()

        if self.model_path is None and self.network is not None:
            insightface_pytorch_model = pytorch_model_loader.load_insightface_pytorch_model(model_name=self.network,
                                                                                            pytorch_weight_path=self.weight_path)
            print('Model Loaded Successfully...')
            self.model=insightface_pytorch_model
            return insightface_pytorch_model

        else:
            print('Error Loading Model')
            return None


    def load_image(self,image_in):
        image = ImageLoader.loader(image_path=image_in,
                                   convert_color=cv2.COLOR_BGR2RGB,
                                   transpose=(2, 0, 1),
                                   dtype=np.float32)
        if torch.cuda.is_available():
            image = torch.Tensor(image).cuda()
            #print('Runing on CUDA Device')
        else:
            image = torch.Tensor(image)
            #print('Runing on CPU ')
        image.div_(255).sub_(0.5).div_(0.5)

        return image


    def get_feature(self,image_in):
        image = self.load_image(image_in)
        feature = self.model(image)
        feature = feature.cpu().detach().numpy()
        embd = feature[0]
        embd = embd/np.linalg.norm(embd)

        return embd