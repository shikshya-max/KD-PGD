# TeachMeToTrick: Transferable PGD Attacks via Multi-Teacher Knowledge Distillation

This repository implements a novel framework for generating transferable **black-box adversarial attacks** using **multi-teacher knowledge distillation (KD)**. A lightweight student model is trained using two powerful teacher networks, and adversarial examples are crafted using **PGD**, **FG**, and **FGS**. These adversarial samples are then evaluated against a **black-box target model** to assess their transferability.

---

## Project Overview

### Problem

How can we generate **highly transferable adversarial examples** for black-box models without direct access to their parameters?

### Solution

Train a student model through **knowledge distillation from two heterogeneous teacher models** (ResNet-50 and DenseNet-161), and use that student to craft attacks (FG, FGS, PGD) that are **effective and computationally efficient**.

---

## Methodology

### Multi-Teacher KD Strategies

- **Curriculum-Based Switching**: The student alternates between teachers every 4 epochs.
- **Joint Optimization**: The student learns from both teachers simultaneously using averaged soft losses.

### Adversarial Attacks

Implemented attacks:
- **FG** – Fast Gradient (L2)
- **FGS** – Fast Gradient Sign (L∞)
- **PGD** – Projected Gradient Descent (iterative, L∞)

These are applied to the student, and tested on a black-box target: **GoogLeNet**.

---

## Key Results (CIFAR-10)
We chose the hyperparameters of the attack methods to maintain an RMSD of **25 ±1**. The attacks were generated on the **test images (N = 10,000)** across all 10 classes and subsequently used to attack the **black-box GoogLeNet model**. Additionally, we limit thenumber of iterations to generate PGD attacks to 10 with a **batch size of 150** and recorded the total time taken for the entire test set. 
| Type        | Attacker Model                          | RMSD  | FG   | FGS  | PGD  | PGD Time (s) |
|-------------|------------------------------------------|-------|------|------|------|--------------|
| Self-Attack | GoogLeNet (Blackbox)                     | 24.48 | 0.80 | 0.88 | 1.00 | 176.47       |
| Baselines   | ResNet-50 (Teacher 1)                    | 24.49 | 0.69 | 0.78 | 0.93 | 69.60        |
| Baselines   | DenseNet-151 (Teacher 2)                 | 24.48 | 0.67 | 0.76 | 0.91 | 139.58       |
| Baselines   | **Ensemble (ResNet-50 & DenseNet-151)** | 24.48 | 0.69 | 0.77 | **0.96** | 201.21  |
| Students    | **Curriculum Trained (Type 1)**          | 24.56 | **0.78** | **0.86** | **0.95** | **33.01** |
| Students    | Jointly Trained (Type 2)                 | 24.55 | 0.73 | 0.83 | 0.93 | 32.03        |

Our proposed student model with **curriculum training** performs comparably to the Ensemble baseline (95% for PGD). Most notably, the time taken to generate 10,000 PGD attacks for our method took only 33 seconds.

---

### Requirements

- Python 3.8+
- PyTorch
- NumPy, tqdm, torchvision
- Optional: Captum (for Grad-CAM)

### Instructions for setup:

1. run `get_data.py` to download and setup CIFAR-10.
2. Download pretrained weights for teachers and blackbox from [here](https://drive.usercontent.google.com/download?id=17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq&export=download&authuser=0). Extract zip inside `models` folder.
3. run `test_models.py` to check if the weights have been loaded.

### Instructions for training:

-   run `train_{"type"}.py` with selected args to train the student with type={"type"}

The trained weights for the students models are provided [here](https://drive.google.com/drive/folders/1PEUiJuVprx271w_Pno4J-onHww0CxH1A?usp=sharing).
Note. Experiment 1 uses lower max lr = 1e-3, and Experiment 2 uses higher max lr = 1e-2. The main results presented in the paper are from Experiment 2.

Checkpoints are saved as 'stu_resnet18\_{'type'}\_a{'alpha'}\_t\_{'tau'}.cpt'. Types = {""} for Curricula based student (Type 1) and {"multiple\*"} for student trained jointly with multiple teachers.

### Model Metrics

-   ResNet50 (Teacher 1): Test=91.39%
-   DenseNet161 (Teacher 2): Test=92.21%
-   GoogLeNet (BlackBox Model): Test=91.26%
-   ResNet18 (Pretrained): Test=90.1%
-   ResNet18 (Sid Trained): Test=87.27%



