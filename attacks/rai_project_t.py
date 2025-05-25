import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

def get_loaders_with_val(batch_size=128, num_workers=2, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_resnet50(pretrained=True, num_classes=10):
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class TeacherModel:
    def __init__(self, device='cuda', pretrained=True, num_classes=10):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = get_resnet50(pretrained=pretrained, num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def finetune(self, train_loader, val_loader=None, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            correct, total, loss_epoch = 0, 0, 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                loss_epoch += loss.item()

            self.scheduler.step()
            print(f"[Epoch {epoch+1}] Loss: {loss_epoch/len(train_loader):.3f} | Train Acc: {100.*correct/total:.2f}%")

            if val_loader:
                val_acc = self.evaluate(val_loader)
                print(f"           → Val Acc: {val_acc:.2f}%")

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100. * correct / total

    def generate_fgsm(self, inputs, targets, epsilon=8/255):
        inputs = inputs.clone().detach().to(self.device).requires_grad_(True)
        targets = targets.clone().detach().to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.model.zero_grad()
        loss.backward()
        adv = inputs + epsilon * inputs.grad.sign()
        return torch.clamp(adv, 0, 1).detach()

    def generate_pgd(self, inputs, targets, epsilon=8/255, alpha=2/255, steps=10):
        inputs = inputs.clone().detach().to(self.device)
        targets = targets.clone().detach().to(self.device)
        orig = inputs.clone().detach()
        adv = inputs + torch.empty_like(inputs).uniform_(-epsilon, epsilon)
        adv = torch.clamp(adv, 0, 1)

        for _ in range(steps):
            adv.requires_grad = True
            loss = self.criterion(self.model(adv), targets)
            self.model.zero_grad()
            loss.backward()
            adv_grad = adv.grad.sign()
            adv = adv.detach() + alpha * adv_grad
            delta = torch.clamp(adv - orig, -epsilon, epsilon)
            adv = torch.clamp(orig + delta, 0, 1)

        return adv.detach()

    def visualize_adversarial(self, test_loader, epsilon=8/255):
        inputs, targets = next(iter(test_loader))
        inputs, targets = inputs[:5].to(self.device), targets[:5].to(self.device)

        with torch.no_grad():
            outputs_clean = self.model(inputs)
        fgsm_inputs = self.generate_fgsm(inputs, targets, epsilon)
        pgd_inputs = self.generate_pgd(inputs, targets, epsilon)

        with torch.no_grad():
            fgsm_preds = self.model(fgsm_inputs).argmax(dim=1)
            pgd_preds = self.model(pgd_inputs).argmax(dim=1)

        def imshow(img, title):
            img = img.cpu().numpy().transpose((1, 2, 0))
            img = (img * 0.5) + 0.5
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')

        plt.figure(figsize=(15, 5))
        for i in range(5):
            plt.subplot(3, 5, i+1)
            imshow(inputs[i], f"Clean: {CIFAR_CLASSES[outputs_clean[i].argmax()]}")
            plt.subplot(3, 5, i+6)
            imshow(fgsm_inputs[i], f"FGSM: {CIFAR_CLASSES[fgsm_preds[i]]}")
            plt.subplot(3, 5, i+11)
            imshow(pgd_inputs[i], f"PGD: {CIFAR_CLASSES[pgd_preds[i]]}")
        plt.tight_layout()
        plt.show()

train_loader, val_loader, test_loader = get_loaders_with_val()

teacher = TeacherModel(device='cuda', pretrained=True)
print("Initial Test Accuracy:", teacher.evaluate(test_loader))

# Fine-tune and monitoring validation set accuracy
teacher.finetune(train_loader, val_loader=val_loader, epochs=3)

val_acc = teacher.evaluate(val_loader)
test_acc = teacher.evaluate(test_loader)

print(f"Final Validation Accuracy: {val_acc:.2f}%")
print(f"Final Test Accuracy: {test_acc:.2f}%")

# Visualize clean vs. adversarial predictions
teacher.visualize_adversarial(test_loader)

