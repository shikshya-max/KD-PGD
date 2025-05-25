from models import googlenet
import torch

'''returns the "black-box" GoogLeNet Model'''
def get_blackbox():
    return googlenet.googlenet(True)
    