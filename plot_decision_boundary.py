'''
Plot decision boundaries of multiple models for a given image.
'''
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import Dataset, Subset
from utils.get_data import get_loaders, classes, inv_normalize, normalize
import random
from models.teachers import get_teachers
from models.student import get_student
from models.blackbox import get_blackbox
from torchvision import transforms
import torchvision.transforms.functional as TTF
from tqdm import tqdm
from itertools import product
import os
from PIL import Image
# setup seeds and make deterministic
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def scale(img_tensor, size=(200, 200)):
    return TTF.resize(img_tensor, size, interpolation=TTF.InterpolationMode.BICUBIC)

def get_selected_images(cat_idx, plane_idx):
    # Note we are using test set as the validation set, so val --> test
    _, val_loader = get_loaders(2, 3)
    val_dataset : Dataset = val_loader.dataset
    labels  = val_dataset.targets
    dataset_cat_idx = np.where(labels == cat_idx)[0]
    dataset_plane_idx = np.where(labels == plane_idx)[0]
    cat_subset = Subset(val_dataset, dataset_cat_idx)
    cat_images = [cat_img for cat_img, _ in cat_subset]
    plane_subset = Subset(val_dataset, dataset_plane_idx)
    plane_images = [plane_img for plane_img, _ in plane_subset]
    # select a good looking image for the 2 classes
    sel_cat_img = cat_images[30]
    # sel_cat_img = cat_images[499]
    sel_plane_img = plane_images[6]
    return sel_cat_img.unsqueeze(0), sel_plane_img.unsqueeze(0)

def generate_adversarial_direction(model, image, label, device):
    """
    Generate a normalized FG adversarial direction.
    """
    model = model.to(device)
    image = image.to(device)
    x = image.detach().requires_grad_()
    output = model(x)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    grad = x.grad.detach()
    # direction = grad / torch.norm(grad)
    grad_flat = grad.view(grad.size(0), -1)  # (B, C*H*W)
    norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)  # (B, 1)
    norm = norm.view(grad.size(0), 1, 1, 1) # (B, 1, 1, 1)
    direction = grad / norm
    return direction.detach()

def orthogonalize_direction(base_dir, new_dir):
    """
    Make new_dir orthogonal to base_dir using Gram-Schmidt process.
    """
    # proj = (torch.sum(new_dir * base_dir) / torch.sum(base_dir * base_dir)) * base_dir
    # orth_dir = new_dir - proj
    orth_dir = new_dir - (new_dir * base_dir).sum() * base_dir
    return orth_dir / torch.norm(orth_dir)

def plot_boundary(img: torch.Tensor, 
                  true_label: int,
                  models_dict: dict[str, torch.nn.Module],
                  models_colors: dict[str, str],
                  save_loc,
                  adversarial_dir_model=get_blackbox(),
                  max_range=60,
                  max_steps=200,
                  batch_size=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get directions
    label = torch.tensor([true_label], dtype=torch.int64, device=device)
    dir1 = generate_adversarial_direction(adversarial_dir_model, img.clone(), label, device)
    random_noise = torch.randn_like(img).to(device)
    dir2 = orthogonalize_direction(dir1, random_noise)
    # setup grid
    xs = np.linspace(-max_range, max_range, max_steps)
    ys = np.linspace(-max_range, max_range, max_steps)
    X, Y = np.meshgrid(xs, ys)
    # precompute all perturbations in batch
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    perturbations = torch.stack([
        x * dir2 + y * dir1
        for x, y in zip(X_flat, Y_flat)
    ]).squeeze(1) # (num_points, C, H, W)
    img_base = inv_normalize(img.detach().to(device))  # (1, C, H, W)
    perturbed_imgs = img_base + perturbations
    perturbed_imgs = torch.clamp(perturbed_imgs, 0, 1)
    perturbed_imgs = normalize(perturbed_imgs)  # still (num_points, C, H, W)

    fig, ax = plt.subplots()

    for name, model in models_dict.items():
        if type(model) != list:
            model = [model]
        for i in range(len(model)):
            model[i] = model[i].eval().to(device)

        print(f"Evaluating model: {name}")

        preds = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, perturbed_imgs.size(0), batch_size), desc=f"{name} batches"):
                end_idx = start_idx + batch_size
                batch = perturbed_imgs[start_idx:end_idx].to(device)
                outputs = 0
                for mod in model:
                    outputs += mod(batch)
                outputs = outputs/len(model)
                batch_preds = torch.argmax(outputs, dim=1)
                preds.append(batch_preds)

        preds = torch.cat(preds)  # (num_points,)
        correct_preds = (preds == true_label).float().cpu().numpy()

        # reshape to (max_steps, max_steps)
        region = correct_preds.reshape(X.shape)

        # plot contour
        ax.contourf(X, Y, region, levels=[0.5, 1.1], colors=[models_colors[name]], alpha=0.2)
        ax.contour(X, Y, region, levels=[0.5], colors=[models_colors[name]], linewidths=2.0)

    # plot settings
    legend_lines = [Line2D([0], [0], color=color, lw=2, label=name) 
                    for name, color in models_colors.items()]
    ax.legend(handles=legend_lines, loc=2)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_title('Decision Boundary Visualization')
    ax.set_xlabel('BlackBox Adversarial Dir (pixels)')
    ax.set_ylabel('Random Orthogonal Dir (pixels)')
    fig.savefig(save_loc)
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    os.makedirs('figs/', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # plane and cat :) for images
    cat_idx = np.argwhere(np.array(classes) == 'cat')[0,0]
    plane_idx = np.argwhere(np.array(classes) == 'plane')[0,0]
    dog_idx = np.argwhere(np.array(classes) == 'dog')[0,0]
    print(f"cat index={cat_idx}, airplane index={plane_idx}")
    # sel_cat_img, sel_plane_img = get_selected_images(cat_idx, plane_idx)
    sel_cat_img, sel_plane_img = get_selected_images(cat_idx, plane_idx)
    
    # load in separate image
    transform_for_model = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    # img_real = Image.open('sample_images/mochi.jpg').convert('RGB')
    # # pass normalized 32x32 to the model to generate adv example 
    # img_32 = scale(img_real, size=(32,32))
    # img = transform_for_model(img_32).unsqueeze(0)
    
    resnet_t, densenet_t = get_teachers()
    blackbox =  get_blackbox()
    
    student =  get_student(pretrained=False) # if we are loading a student --> pretrained=False
    # Best Results Model
    cpt = f"checkpoints_experiment_2/stu_resnet18_a_03_t_1.cpt"
    student.load_state_dict(torch.load(cpt))
    
    models_dict = {
        'ResNet-50(Teacher-1)': resnet_t,
        'DenseNet-161(Teacher-2)': densenet_t,
        'GoogLeNet(Blackbox)': blackbox,
        'Ensemble(T1&T2)': [resnet_t, blackbox],
        'Student': student
    }
    
    models_colors = {
        'ResNet-50(Teacher-1)': 'blue',
        'DenseNet-161(Teacher-2)': 'green',
        'GoogLeNet(Blackbox)': 'orange',
        'Ensemble(T1&T2)': 'purple',
        'Student': 'red'
        # red for student
    }

    # save_loc = 'figs/boundary_mochi_zoom_out.png'
    save_loc = 'figs/test_zoom_in_cat.png'
    # save_loc = 'figs/test.png'
    plot_boundary(sel_cat_img, cat_idx, models_dict, models_colors, save_loc, adversarial_dir_model=blackbox,
                  max_range=10)
    


