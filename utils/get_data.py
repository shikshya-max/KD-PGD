import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch
import numpy as np

# mean and sd from https://github.com/huyvnphan/PyTorch_CIFAR10/blob/641cac24371b17052b9bb6e56af1c83b5e97cd7f/data.py#L16

mean=[0.4914, 0.4822, 0.4465]
std=[0.2471, 0.2435, 0.2616]

normalize = transforms.Normalize(mean,std)

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std])


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get the ciphar-10 dataset
def get_loaders(batch_size, num_workers, val_batch_size=256):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=val_batch_size, shuffle=False,
            num_workers=2, pin_memory=True)
    return train_loader, val_loader

if __name__ == "__main__":
    # try to display some random images
    train_dl, val_dl = get_loaders(32, 2)
    dataiter = iter(train_dl)
    images, labels = next(dataiter)
    images = inv_normalize(images)
    images = images.clamp(0, 1)
    collage = torchvision.utils.make_grid(images[:10], 2)
    collage = collage.numpy()
    print(np.max(collage), np.min(collage))
    plt.imshow(np.transpose(collage, (1, 2, 0)))
    plt.show()