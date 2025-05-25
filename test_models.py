'''
This file generates test loss and accuracy for all the used models.
'''
from utils.get_data import get_loaders
from utils.utils import get_model_metrics
import torch
from models.teachers import get_teachers
from models.student import get_student
from models.blackbox import get_blackbox


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check train acc for both teachers
    
    train_dl, test_dl = get_loaders(32, 4)
    resnet_t, densenet_t = get_teachers()
    print("ResNet50 Test Acc:", get_model_metrics(resnet_t, test_dl, device=device))
    print("DenseNet161 Test Acc:", get_model_metrics(densenet_t, test_dl, device=device))
    
    blackbox =  get_blackbox()
    print("GoogLeNet Test Acc:", get_model_metrics(blackbox, test_dl, device=device))

    student =  get_student(pretrained=False) # if we are loading a student --> pretrained=False
    # UPDATE AS NEEDED
    cpt = "checkpoints/stu_resnet18_multiple_a_03_t_1.cpt"
    cpt = "checkpoints/resnet-18-best.cpt"
    
    student.load_state_dict(torch.load(cpt))
    
    print("Resnet18 Test Acc:", get_model_metrics(student, test_dl, device=device))