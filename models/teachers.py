from models import resnet
from models import densenet
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_teachers():
    resnet_t = resnet.resnet50(True) 
    densenet_t = densenet.densenet161(True)
    return resnet_t, densenet_t



    
            

