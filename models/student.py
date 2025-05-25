import torch.nn as nn
import torch
from models.resnet import resnet18, resnet50

'''
Student Network -- Most cases this will be ResNet18
'''
def get_student(type='resnet18', device='cpu', pretrained=False):
    model = None
    if type == 'resnet18':
        model =  resnet18(pretrained)
    elif type == 'resnet50':
        model = resnet50(pretrained)
    else:
        raise NotImplementedError(f"Student {type} has not been configured in student.py")
    return model.to(device)

