import torch
import torch.nn as nn
import  torch.nn.functional as F

# Calculate metrics for the dl
def get_model_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                      criterion=None, teacher=None, device='cpu'):
    acc = 0
    loss = 0
    model = model.to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            out: torch.Tensor = model(images)
            pred = out.argmax(1)
            acc += torch.sum(pred == labels)
            if criterion is not None:
                if teacher is not None:
                    teacher_out = None
                    if issubclass(type(teacher), torch.nn.Module):
                        teacher_out = teacher(images) # single teacher output
                    else:
                        teacher_out = [t(images) for t in teacher] # list of multiple teacher outputs
                    loss += criterion(out, labels, teacher_out)
                else:
                    loss += criterion(out, labels) # no teacher
    acc = acc/len(dataloader.dataset) # per sample acc 
    if criterion is not None:
        loss = loss/len(dataloader) # per sample loss
        loss = loss.item()
    return loss, acc.item()

def get_model_metrics_batch(model: torch.nn.Module, images, adv_images, labels, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    with torch.no_grad():
        images, adv_images, labels = images.to(device), adv_images.to(device), labels.to(device)
        out_real: torch.Tensor = model(images)
        pred_real = out_real.argmax(1)
        acc_real = torch.sum(pred_real == labels)
        out_adv: torch.Tensor = model(adv_images)
        
        pred_adv = out_adv.argmax(1)
        acc_adv = torch.sum(pred_adv == labels)
        delta_acc = acc_real - acc_adv
        print(f"Correct: (R:{acc_real}), (Adv:({acc_adv})) | Delta: {delta_acc} | Num : {images.size(0)}")
        return acc_real, acc_adv, delta_acc
        

# define the loss for distillation
def distillation_loss(out, labels, teacher_logits, alpha = 0, softmax_temp=1):
    """Loss for the project

    Args:
        out (torch.Tensor): Student model output
        labels (torch.Tensor): Ground truth label
        teacher_logits (torch.Tensor): Teacher model output
        alpha (float): Controls the tradeoff between teacher knowledge and student self-learning. Defaults to 0.
        softmax_temp (int, optional): _description_. Defaults to 1.
    Returns:
        loss: returns a*(hard_loss) + (1-a)*soft_loss
    """
    student_log_probs = F.log_softmax(out / softmax_temp, dim=1)
    teacher_probs = F.softmax(teacher_logits / softmax_temp, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (softmax_temp ** 2)
    hard_loss =  nn.CrossEntropyLoss(reduction='mean')(out, labels)
    loss = alpha*(hard_loss) + (1-alpha)*soft_loss
    return loss

# define the loss for distillation
def distillation_loss_multiple(out, labels, teacher_logits:list[torch.Tensor], alpha = 0, softmax_temp=1):
    """Loss for the project

    Args:
        out (torch.Tensor): Student model output
        labels (torch.Tensor): Ground truth label
        teacher_logits (list[torch.Tensor]): Teacher model output
        alpha (float): Controls the tradeoff between teacher knowledge and student self-learning. Defaults to 0.
        softmax_temp (int, optional): _description_. Defaults to 1.
    Returns:
        loss: returns a*(hard_loss) + (1-a)*soft_loss
    """
    student_log_probs = F.log_softmax(out / softmax_temp, dim=1)
    soft_loss = 0
    for teacher_logit in teacher_logits:
        teacher_probs = F.softmax(teacher_logit / softmax_temp, dim=1)
        soft_loss += F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (softmax_temp ** 2)
    soft_loss = soft_loss/len(teacher_logits)
    hard_loss =  nn.CrossEntropyLoss(reduction='mean')(out, labels)
    loss = alpha*(hard_loss) + (1-alpha)*soft_loss
    return loss


# simple class to store dist_loss hparams
class DistillationLoss:
    def __init__(self, alpha = 0, softmax_temp=1, teacher_type='single'):
        """_summary_

        Args:
            alpha (int, optional): _description_. Defaults to 0.
            softmax_temp (int, optional): _description_. Defaults to 1.
            teacher_type (str, optional): _description_. Defaults to 'single'.
        """
        self.alpha = alpha
        self.softmax_temp = softmax_temp
        self.loss = None
        if teacher_type == 'single':
            self.loss = distillation_loss
        elif teacher_type == 'multiple':
            self.loss = distillation_loss_multiple
        else:
            raise NotImplementedError(f"teacher type {teacher_type} not implemented.")

    def __call__(self, out, labels, 
                 teacher_logits:torch.Tensor|list[torch.Tensor]):
        return self.loss(out, labels, teacher_logits, alpha=self.alpha, softmax_temp=self.softmax_temp)