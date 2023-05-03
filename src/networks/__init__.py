from torchvision import models

from .aceto19cnn import Aceto19CNN
from .aceto19mimetic import Aceto19MIMETIC
from .aceto19mimeticp import Aceto19MIMETICP
from .aceto19mimeticpp import Aceto19MIMETICPP
from .aceto19mimetic_no_shared import Aceto19MIMETICNoShared
from .aceto19rnn import Aceto19RNN
from .lenet import LeNet
from .lopez17cnn import Lopez17CNN
from .lopez17cnn_drop import Lopez17CNNDrop
from .lopez17cnnmlp import Lopez17CNNMLP
from .lopez17cnnrnn import Lopez17CNNRNN
from .lopez17rnncnn import Lopez17RNNCNN
from .lopez17rnn import Lopez17RNN
from .resnet32 import resnet32
from .resnet import *
from .vggnet import VggNet
from .wang17 import Wang17
from .shallow_autoencoder import ShallowAutoencoder
from .rnn_autoencoder import RecurrentAutoencoder
from .oleksii17_autoencoder import Oleksii17Autoencoder
#from .yang21cnn import Yang21CNN
#from .chen21rnn import Chen21RNN

# available torchvision models
tvmodels = ['alexnet',
            'densenet121', 'densenet169', 'densenet201', 'densenet161',
            'googlenet',
            'inception_v3',
            'mobilenet_v2',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
            'squeezenet1_0', 'squeezenet1_1',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
            'wide_resnet50_2', 'wide_resnet101_2'
            ]

netmodels = ['Wang17', 'Lopez17CNN', 'Lopez17RNN', 'Lopez17CNNRNN', 'Lopez17RNNCNN', 'Aceto19CNN', 'Aceto19RNN',
             'Lopez17CNNDrop', 'Lopez17CNNMLP', 'Aceto19MIMETIC', 'Aceto19MIMETICP', 'Aceto19MIMETICPP', 
             'Aceto19MIMETICNoShared', 'Yang21CNN', 'Chen21RNN']

autoencoders = ['ShallowAutoencoder', 'RecurrentAutoencoder', 'Oleksii17Autoencoder']

allmodels = tvmodels + netmodels + autoencoders + ['resnet32', 'LeNet', 'VggNet']

def set_tvmodel_head_var(model):
    if type(model) == models.AlexNet:
        model.head_var = 'classifier'
    elif type(model) == models.DenseNet:
        model.head_var = 'classifier'
    elif type(model) == models.Inception3:
        model.head_var = 'fc'
    elif type(model) == models.ResNet:
        model.head_var = 'fc'
    elif type(model) == models.VGG:
        model.head_var = 'classifier'
    elif type(model) == models.GoogLeNet:
        model.head_var = 'fc'
    elif type(model) == models.MobileNetV2:
        model.head_var = 'classifier'
    elif type(model) == models.ShuffleNetV2:
        model.head_var = 'fc'
    elif type(model) == models.SqueezeNet:
        model.head_var = 'classifier'
    else:
        raise ModuleNotFoundError
