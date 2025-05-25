import torch
import torchvision
from utils.get_data import inv_normalize, normalize
import torch.nn.functional as F
from tqdm import tqdm

def get_grad(model, images, labels, reduction='mean'):
    x = images.detach().requires_grad_()
    
    if issubclass(type(model), torch.nn.Module): # not ensemble = nn.Module
        model = [model]
    loss = 0
    for a_model in model: 
        output = a_model(x)
        loss += F.cross_entropy(output, labels, reduction=reduction)
        a_model.zero_grad()
    loss = loss/len(model)
    loss.backward()
    grad = x.grad.detach()
    return grad



def fgsm_attack(model, images, labels, epsilon=9/255):
    grad = get_grad(model, images, labels)
    # Collect the element-wise sign of the data gradient
    sign_data_grad = grad.sign()
    perturbed_images = images + epsilon * sign_data_grad # TODO
    perturbed_images = inv_normalize(perturbed_images.detach())
    # Adding clipping to maintain [0,1] range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    perturbed_images = normalize(perturbed_images)
    return perturbed_images

def fgm_attack(model, images, labels, epsilon=9/255):
    grad = get_grad(model, images, labels, reduction='sum')
    grad_flat = grad.view(grad.size(0), -1)  # (B, C*H*W)
    norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)  # (B, 1)
    norm = norm.view(grad.size(0), 1, 1, 1) # (B, 1, 1, 1)
    norm_grad = grad / norm
    perturbed_images = images + epsilon * norm_grad
    perturbed_images = inv_normalize(perturbed_images) #TODO
    # Adding clipping to maintain [0,1] range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    perturbed_images = normalize(perturbed_images)
    return perturbed_images


def pgd_attack(model, images, labels,
               epsilon=9/255, alpha=2/255, num_steps=20):
    
    if type(model) == list: # if ensemble
        for i in range(len(model)):
            model[i] = model[i].eval()
    else: 
        model = model.eval()

    images = images.clone().detach()
    labels = labels.clone().detach()    
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(inv_normalize(adv_images), 0, 1)
    adv_images = normalize(adv_images).detach()

    for _ in range(num_steps):
        grad = get_grad(model, adv_images, labels)
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon) # project back to eps-ball
        adv_images = inv_normalize(images + delta)
        adv_images = torch.clamp(adv_images, 0, 1)
        adv_images = normalize(adv_images).detach()

    return adv_images

