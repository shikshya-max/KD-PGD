import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.lines import Line2D
from torch.utils.data import Dataset, Subset, DataLoader
from utils.get_data import get_loaders, classes, inv_normalize, normalize
from utils.utils import get_model_metrics
from models.teachers import get_teachers
from models.student import get_student
from models.blackbox import get_blackbox
from attacks.attacks import fgm_attack, fgsm_attack, pgd_attack
import torchvision

def rmsd_per_sample(real_imgs, adv_imgs):
    B, C, H, W = real_imgs.shape
    N = C * H * W
    real_imgs = inv_normalize(real_imgs)*255.0 # Converting to 0-255 Range
    adv_imgs = inv_normalize(adv_imgs)*255.0 # Converting to 0-255 Range
    
    diff_squared = (real_imgs - adv_imgs) ** 2
    sum_diff_squared = diff_squared.view(B, -1).sum(dim=1)
    return torch.sqrt(sum_diff_squared/ N)


def get_class_images(val_loader, class_idx):
    # Note we are using test set as the validation set, so val --> test
    val_dataset : Dataset = val_loader.dataset
    labels  = val_dataset.targets
    dataset_idx = np.where(labels == class_idx)[0]
    subset = Subset(val_dataset, dataset_idx)
    return subset

def display_examples(imgs, adv_imgs):
    fig, axes = plt.subplots(1, 2)
    collage_real = torchvision.utils.make_grid(inv_normalize(imgs[:9]), 3)
    collage_real = collage_real.cpu().numpy()
    axes[0].imshow(np.transpose(collage_real, (1, 2, 0)))
    axes[0].set_title("Real")
    collage_adv = torchvision.utils.make_grid(inv_normalize(adv_imgs[:9]), 3)
    collage_adv = collage_adv.cpu().numpy()
    axes[1].imshow(np.transpose(collage_adv, (1, 2, 0)))
    axes[1].set_title("Adversarial")
    plt.show()

# attack an entire class from the validation/test set
def attack_class_val(attack_model, target_model, val_loader, class_idx, device, 
                     epsilon=9/255, alpha=2/255, num_steps=10, attack_type='fgsm',
                     display_images=False):
    if type(attack_model) == list:
        for i in range(len(attack_model)):
            attack_model[i] = attack_model[i].eval() # sid is paranoid
    
    target_model = target_model.eval()
    subset = get_class_images(val_loader, class_idx)
    class_dl = torch.utils.data.DataLoader(
            subset,
            # batch_size=150, shuffle=False,
            batch_size=400, shuffle=False,
            num_workers=1, pin_memory=True)
    real_correct = 0
    adv_correct = 0
    total = 0
    total_rmsd_list = []
    for (imgs, labels) in class_dl:   
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = None
        if attack_type == "fgsm":
            adv_imgs = fgsm_attack(attack_model, imgs.clone().detach(), labels, epsilon=epsilon)
        elif attack_type == "fgm":
            adv_imgs = fgm_attack(attack_model, imgs.clone().detach(), labels, epsilon=epsilon)
        elif attack_type == "pgd":
            adv_imgs = pgd_attack(attack_model, imgs.clone().detach(), labels, epsilon=epsilon, 
                                  alpha=alpha, num_steps=num_steps)
        else:
            raise NotImplementedError(f"attack_type= {attack_type} not supported")
        if display_images:
            display_images = False # once per class
            display_examples(imgs, adv_imgs)
        with torch.no_grad():
            real_out = target_model(imgs)
            real_pred = real_out.argmax(1)
            real_correct_batch = torch.sum(real_pred == labels).item()
            real_correct += real_correct_batch

            adv_out = target_model(adv_imgs)
            adv_pred = adv_out.argmax(1)
            adv_correct_batch = torch.sum(adv_pred == labels).item()
            adv_correct += adv_correct_batch
            total += imgs.shape[0]
            batch_rmsd = rmsd_per_sample(imgs, adv_imgs) # (B,)
            total_rmsd_list.append(batch_rmsd)
            
    all_rmsd = torch.cat(total_rmsd_list)
    class_rmsd = all_rmsd.mean().item()
    return real_correct, adv_correct, total, class_rmsd
            
if __name__ == "__main__":
    _, val_loader = get_loaders(2, 1, val_batch_size=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blackbox = get_blackbox().eval()
    resnet_t, densenet_t = get_teachers()
    resnet_t = resnet_t.eval()
    densenet_t = densenet_t.eval()
    # # PARAMS
    student =  get_student(pretrained=False)
    alpha = "03"
    temp = "5"
    stu_type = "_multiple" # "" for type 1, "_multiple" for type 2
    # stu_type = "" # "" for type 1, "_multiple" for type 2
    cpt = f"checkpoints_experiment_1/stu_resnet18{stu_type}_a_{alpha}_t_{temp}.cpt"
    # cpt = f"checkpoints_experiment_2/stu_resnet18{stu_type}_a_{alpha}_t_{temp}.cpt"
    student.load_state_dict(torch.load(cpt))
    student = student.eval()
    # ARGS
    attack_model = student # which model to use to generate attacks, pass list for ensemble
    target_model = blackbox # which model to attack
    target_model = target_model.to(device)
    # --TYPE--
    # attack_type = 'fgsm' # fgm (l2 norm) or fgsm  or pgd
    # attack_type = 'fgm' # fgm (l2 norm) or fgsm  or pgd
    attack_type = 'pgd' # fgm (l2 norm) or fgsm  or pgd
    # --EPS-- selected params to maintain 25 rmsd across experiments
    epsilon = None
    if attack_type == 'fgm':
        epsilon = 23.0
    elif attack_type == 'fgsm':
        epsilon = 100/255
    elif attack_type == 'pgd':
        epsilon = 175/255 # for pgd
    
    
    # --OTHER--
    num_steps = 10 # pgd num steps
    alpha=20/255 # pgd step size
    verbose = False # output perclass info?
    # verbose = True # output perclass info?
    display_images = False # show images per batch?
    # display_images = True # show images per batch?
    
    if type(attack_model) == list:
        for i in range(len(attack_model)):
            attack_model[i] = attack_model[i].to(device)
    else:
        attack_model = attack_model.to(device)
    corr = 0
    total_all = 0
    total_rmsd = 0 
    print("Running Attack", attack_type, 'e =', epsilon)
    for class_name in list(classes):
    # for class_name in ['cat']:
        class_idx = np.argwhere(np.array(classes) == class_name)[0,0]
        real_correct, adv_correct, total, class_rmsd = attack_class_val(attack_model, target_model, val_loader, 
                                                                        class_idx, device, num_steps=num_steps, 
                                                                        alpha=alpha,
                                                                        epsilon=epsilon, attack_type=attack_type,
                                                                        display_images=display_images)
        corr += adv_correct # number of adv images the target model predicted correct, lower is better
        total_all += total # total number in the class
        total_rmsd += class_rmsd*1000  # 1000 images per class; class_rmsd is average per images
        if verbose:
            print(class_name, 
                f": RMSD: {class_rmsd:.3f},",
                f"Real%: ({real_correct/total:.3f}),",
                f"Attack%: ({adv_correct/total:.3f}),",
                f"ASR%: ({1 - (adv_correct/total):.3f})")
    # Total Test Set Stats
    print(f"Test RMSD {total_rmsd/total_all:.3f}")
    print("Adv Succ Count", total_all-corr, "Total Images", total_all)
    print("Test Set ASR:", 1 - (corr/total_all))