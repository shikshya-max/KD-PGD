'''
This file handles the training of the student model
'''
import optuna
import torch.nn as nn
import torch
from models.teachers import get_teachers
from models.student import get_student
from utils.get_data import get_loaders
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils import get_model_metrics, distillation_loss
import torch.optim as optim

TUNE_EPOCHS = 18

def get_h_params(trial, student_model, max_tune_epochs):
    # setup optimizers and scheduler
    learning_rate_init = trial.suggest_float(
        "learning_rate_init", 1e-4, 1e-2
    )
    switch_epoch = trial.suggest_int(
        "switch_epoch", 2, 10, step=1
    )
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    optimizer = getattr(optim, optimizer_name)(student_model.parameters(), lr=learning_rate_init)
    scheduler_name = trial.suggest_categorical("scheduler", ["ReduceLROnPlateau", "CosineAnnealingLR"])
    scheduler = None
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 'max') # trying to max acc
    else:
        scheduler = CosineAnnealingLR(optimizer, max_tune_epochs)
    return switch_epoch, optimizer, scheduler


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get student and teacher models
    t1, t2 = get_teachers()
    s1 = get_student()
    t1, t2, s1 = t1.to(device), t2.to(device), s1.to(device) 
    s1 = s1.train(True)
    t1, t2 = t1.eval(), t2.eval()
    # get ciphar-10 data
    train_dl, val_dl = get_loaders(256, 2)

    switch_epoch, optimizer, scheduler = get_h_params(trial, s1, TUNE_EPOCHS)
    switch_counter = switch_epoch # switch teacher every N epochs
    t1_select = False
    teacher = t2
    epoch_val_acc = -1
    for i in range(1, TUNE_EPOCHS+1):
        # lets assume for now simple schedule
        # teachers switch every 3 epochs
        # TODO move this logic to a teacher_schedule function
        switch_counter -= 1
        if switch_counter == 0:
            switch_counter = switch_epoch
            if t1_select: # t1 was previous teacher
                teacher = t2
                t1_select = False
            else: # t2 was previous teacher
                teacher = t1
                t1_select = True
        _, acc = train_epoch(s1, teacher, optimizer, train_dl, device)
        _, epoch_val_acc = get_model_metrics(s1, val_dl, device=device)
        scheduler_name = type(scheduler).__name__
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(acc)
        elif scheduler_name == "CosineAnnealingLR":
            scheduler.step(i)
        else:
            raise NotImplementedError("Selected scheduler not supported in train_epoch.")
        trial.report(epoch_val_acc, i)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return epoch_val_acc


# trains the model for a single epoch and returns loss/acc
def train_epoch(student: nn.Module, teacher: nn.Module, 
                optimizer, train_dl: torch.utils.data.DataLoader, 
                device='cpu'):
    student.train()
    teacher.eval()
    total_loss = 0
    total_correct = 0
    for idx, (imgs, labels) in enumerate(train_dl):
        imgs, labels = imgs.to(device), labels.to(device)
        t_logits = None
        with torch.no_grad():
            t_logits = teacher(imgs)
        s_logits = student(imgs)
        # calculate acc and loss
        loss = distillation_loss(s_logits, labels, t_logits, 0, 1)
        total_correct += torch.sum(s_logits.argmax(1) == labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    num_samples = len(train_dl.dataset)
    num_batches = len(train_dl)
    loss, acc = total_loss/num_batches, total_correct/num_samples
    return loss, acc


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="student-1-resnet-18-simple_sch",
        direction='maximize',
        load_if_exists=True
    )
    timeout = 12*60*60 # 12 hrs
    study.optimize(objective, n_trials=50, timeout=timeout, n_jobs=5)
    print(f"Best value: {study.best_value} (params: {study.best_params})") 
    # So far: 0.002 with val acc=0.817, 15 epochs