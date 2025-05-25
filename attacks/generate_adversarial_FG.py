import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.student import get_student

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load a student model
model = get_student(pretrained=False, device=device)  # Recreate the same model architecture
model.load_state_dict(torch.load('student_model.pth', map_location=device))
model.to(device)


model.eval()

# FGSM function to generate adversarial examples
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

def fgvm_attack(image, epsilon, data_grad):
    # Use the gradient values directly (instead of sign)
    perturbed_image = image + epsilon * data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

def generate_adversarial_examples(loader, model, epsilon, attack_type='fgsm'):
    correct = 0
    adv_examples = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Forward pass
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, don't bother attacking
        if init_pred.item() != target.item():
            continue

        # Calculate loss
        loss = F.cross_entropy(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients
        loss.backward()

        # Collect gradient
        data_grad = data.grad.data

        # Call attack function
        if attack_type == 'fgsm':
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack_type == 'fgvm':
            perturbed_data = fgvm_attack(data, epsilon, data_grad)
        else:
            raise ValueError("Attack type not supported")

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
        else:
            # Save some examples for visualization
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy
    final_acc = correct / float(len(loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(loader)} = {final_acc}")

    return final_acc, adv_examples

epsilons = [0, 0.01, 0.05, 0.1, 0.15, 0.2]
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = generate_adversarial_examples(testloader, model, eps, attack_type='fgsm')
    accuracies.append(acc)
    examples.append(ex)
    print("Model on:", next(model.parameters()).device)

