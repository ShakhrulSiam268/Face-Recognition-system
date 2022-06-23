import sys
import torch
from torchsummary import summary
import pkg_resources
import time
import traceback
from resnet100.backbones import get_model
import resnet100.backbones


def exception_printer(error_message):

    exception_message = traceback.format_exc(limit=2)

    print('\n' + str(error_message))
    print(exception_message)


class PyTorchModelLoader:
    def __init__(self):
        self.init_device()

        print('*********** PyTorch Model Loader ***********')
        print('torch version: ', torch.__version__)
        print('torch cuda is available: ', torch.cuda.is_available())
        print('torch device: ', self.device)
        print('torchsummary version: ', pkg_resources.get_distribution('torchsummary').version)
        


    def init_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        


    def load_insightface_pytorch_model(self, model_name=None, pytorch_model_path=None, pytorch_weight_path=None, input_shape=(3, 112, 112), train=False):
        start_time = time.time()

        if pytorch_model_path is not None:
            print('\nStarting load insightface pytorch model \'' + str(pytorch_model_path) + '\'...')

            try:
                self.pytorch_model = torch.load(pytorch_model_path, map_location=self.device)

            except Exception as ex:
                exception_printer('Load pytorch model failed.')
                return None

        elif model_name is not None and pytorch_weight_path is not None:
            print('\nStarting load insightface pytorch model name: ' + str(model_name) + ', weight: \'' + str(pytorch_weight_path) + '\'...')

            try:
                self.pytorch_model = get_model(name=model_name)
                self.pytorch_model.load_state_dict(torch.load(pytorch_weight_path,map_location=self.device))

            except Exception as ex:
                exception_printer('Load pytorch weight failed.')
                return None

        self.pytorch_model.to(device=self.device)
        self.pytorch_model.train(train)

        #summary(self.pytorch_model, input_size=input_shape)

        print('Load pytorch model success. Cost time: ' + str(time.time() - start_time) + 's.')
        return self.pytorch_model
