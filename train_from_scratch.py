"""
From-scratch training script for Net_conditional (AZ80 microstructure -> mechanical
properties), built from the two source files in this repo:

  - microstructure-to-properties.ipynb : source of truth for the model architecture,
    EMA class, save_model()/load_model(), class_maker(), noise(), and the test()
    evaluation function (all identical to train.py's copies).
  - train.py (the project's actual training script, referred to as "train.txt" in the
    task spec): source of truth for the training loop, data pipeline, loss weighting,
    and hyperparameters actually used in practice.

Unlike train.py, this script trains from randomly initialized weights instead of
restoring the Mic-Mech-Over-800.pth.tar checkpoint and continuing for 200 more epochs.
See README.md for exactly which values are ported verbatim vs. newly introduced.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import copy
import csv

# =====================================================================================
# CONFIG -- new: all paths/hyperparameters that must be edited per-environment are
# collected here (matching the editable-variable convention already used for
# `external_validation_dir` in microstructure-to-properties.ipynb).
# =====================================================================================

trial = 1

# --- Data / checkpoint directories --------------------------------------------------
# new -- these were hardcoded Kaggle absolute paths in train.py
# (e.g. '/kaggle/input/mic-mech/Mic-Mech/Train-Oversampled'). The dataset now ships
# inside this repo under az80-microstructure-data/, so the defaults below work
# unchanged locally, in Colab, or after `git clone` on Kaggle -- no per-environment
# dataset attachment needed. Override via env vars only if you keep the data
# somewhere else (e.g. a separate Kaggle Dataset mounted at /kaggle/input/...).
whole_dir = os.environ.get("WHOLE_DIR", "az80-microstructure-data/Train-Oversampled")
seen_test_dir = os.environ.get("SEEN_TEST_DIR", "az80-microstructure-data/Test-Seen")
unseen_test_dir = os.environ.get("UNSEEN_TEST_DIR", "az80-microstructure-data/Test-Unseen")
save_dir = os.environ.get("SAVE_DIR", "checkpoints/")
os.makedirs(save_dir, exist_ok=True)

# new -- fail with a clear message instead of torchvision's raw FileNotFoundError
# from deep inside ImageFolder if a directory is missing (e.g. Train-Oversampled
# has not been added to az80-microstructure-data/ yet).
for _name, _path in [("WHOLE_DIR", whole_dir), ("SEEN_TEST_DIR", seen_test_dir),
                      ("UNSEEN_TEST_DIR", unseen_test_dir)]:
    if not os.path.isdir(_path):
        raise FileNotFoundError(
            f"{_name} does not exist: '{_path}'. Set the {_name} environment "
            f"variable to the correct path, or add the missing folder under "
            f"az80-microstructure-data/.")

# new -- train.py loaded a pretrained checkpoint here before training (making it a
# continuation/fine-tuning run). This script trains from scratch, so no checkpoint is
# loaded at start. `resume_dir` is kept only as an optional manual-resume hook: leave
# it as None for a genuine from-scratch run.
resume_dir = os.environ.get("RESUME_DIR", None)

# ported from train.py -- auto cuda/cpu detection, used unchanged
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters -----------------------------------------------------------------
# ported from train.py -- exact values used in practice, do not change without reason
batch_size = 4              # train.py: batch_size = 4 (kept configurable for smaller/larger GPUs)
learning_rate = 3e-5         # train.py: learning_rate = 3e-5
image_shape = (1, 512, 512)  # train.py: image_shape
image_size = 512             # train.py: image_size
magnifications = 4           # train.py: magnifications
embedding_dim = 100          # train.py: embedding_dim
num_classes = 83             # train.py: num_classes
ys_range = 105
uts_range = 125
el_range = 9.25
e_range = 12
k_range = 250.11
n_range = 0.104

# new -- added to convert continuation run into from-scratch training.
# The paper (Section 3.2) states training "was completed after approximately 800
# epochs" with early stopping -- that is the basis for n_epoch below. train.py itself
# only ever ran 200-epoch *continuation* windows on top of an already-trained
# checkpoint, so it has no single "total epochs from scratch" value to port.
n_epoch = 800          # from the paper (~800 epochs to convergence), NOT from train.py
patience = 50          # new -- our own default; not specified in the paper or train.py
min_delta = 0.0        # new -- our own default; minimum improvement to reset patience

# new -- train.py initialized min_loss = 1848.41, a value carried over from a *prior*
# training run's checkpoint. A from-scratch run has no such prior baseline, so we start
# from +inf, matching microstructure-to-properties.ipynb / train.py's own
# `total_loss_min = np.Inf` convention.
min_loss = np.Inf

# =====================================================================================
# ported from train.py -- class/property lookup tables (class_table/seen_table/
# unseen_table) and their label-string dictionaries, used to look up YS/UTS/EL/E/K/N
# for each ImageFolder class index. Copied verbatim.
# =====================================================================================

class_table = torch.tensor([[    0.000,     1.000,     2.000,     3.000,     0.000,     1.000,
             2.000,     3.000,     0.000,     1.000,     2.000,     3.000,
             0.000,     1.000,     3.000,     0.000,     1.000,     2.000,
             3.000,     0.000,     1.000,     2.000,     3.000,     0.000,
             2.000,     0.000,     1.000,     2.000,     3.000,     0.000,
             1.000,     3.000,     0.000,     1.000,     3.000,     0.000,
             1.000,     2.000,     3.000,     0.000,     1.000,     2.000,
             3.000,     0.000,     1.000,     3.000,     0.000,     1.000,
             2.000,     3.000,     0.000,     1.000,     2.000,     3.000,
             0.000,     1.000,     3.000,     0.000,     1.000,     3.000,
             0.000,     1.000,     3.000,     0.000,     1.000,     3.000,
             0.000,     1.000,     2.000,     3.000,     0.000,     1.000,
             2.000,     3.000,     0.000,     1.000,     3.000,     0.000,
             1.000,     3.000,     0.000,     1.000,     3.000],
        [  206.000,   206.000,   206.000,   206.000,   193.000,   193.000,
           193.000,   193.000,   212.000,   212.000,   212.000,   212.000,
           251.000,   251.000,   251.000,   244.000,   244.000,   244.000,
           244.000,   151.000,   151.000,   151.000,   151.000,   148.000,
           148.000,   178.000,   178.000,   178.000,   178.000,   210.000,
           210.000,   210.000,   195.000,   195.000,   195.000,   177.000,
           177.000,   177.000,   177.000,   146.000,   146.000,   146.000,
           146.000,   183.000,   183.000,   183.000,   202.000,   202.000,
           202.000,   202.000,   172.000,   172.000,   172.000,   172.000,
           216.000,   216.000,   216.000,   197.000,   197.000,   197.000,
           227.000,   227.000,   227.000,   230.000,   230.000,   230.000,
           223.000,   223.000,   223.000,   223.000,   198.000,   198.000,
           198.000,   198.000,   183.000,   183.000,   183.000,   216.000,
           216.000,   216.000,   209.000,   209.000,   209.000],
        [  293.000,   293.000,   293.000,   293.000,   284.000,   284.000,
           284.000,   284.000,   289.000,   289.000,   289.000,   289.000,
           330.000,   330.000,   330.000,   309.000,   309.000,   309.000,
           309.000,   230.000,   230.000,   230.000,   230.000,   210.000,
           210.000,   226.000,   226.000,   226.000,   226.000,   271.000,
           271.000,   271.000,   239.000,   239.000,   239.000,   299.000,
           299.000,   299.000,   299.000,   256.000,   256.000,   256.000,
           256.000,   252.000,   252.000,   252.000,   280.000,   280.000,
           280.000,   280.000,   246.000,   246.000,   246.000,   246.000,
           335.000,   335.000,   335.000,   306.000,   306.000,   306.000,
           311.000,   311.000,   311.000,   305.000,   305.000,   305.000,
           302.000,   302.000,   302.000,   302.000,   326.000,   326.000,
           326.000,   326.000,   299.000,   299.000,   299.000,   307.000,
           307.000,   307.000,   305.000,   305.000,   305.000],
        [    5.830,     5.830,     5.830,     5.830,     5.750,     5.750,
             5.750,     5.750,     4.160,     4.160,     4.160,     4.160,
             4.980,     4.980,     4.980,     3.170,     3.170,     3.170,
             3.170,     3.520,     3.520,     3.520,     3.520,     3.490,
             3.490,     2.130,     2.130,     2.130,     2.130,     2.640,
             2.640,     2.640,     1.590,     1.590,     1.590,     4.790,
             4.790,     4.790,     4.790,     5.320,     5.320,     5.320,
             5.320,     2.220,     2.220,     2.220,     4.210,     4.210,
             4.210,     4.210,     1.900,     1.900,     1.900,     1.900,
             6.900,     6.900,     6.900,    10.010,    10.010,    10.010,
             6.310,     6.310,     6.310,     4.670,     4.670,     4.670,
             5.710,     5.710,     5.710,     5.710,     6.390,     6.390,
             6.390,     6.390,    10.840,    10.840,    10.840,     7.240,
             7.240,     7.240,     7.890,     7.890,     7.890],
        [   45.000,    45.000,    45.000,    45.000,    46.000,    46.000,
            46.000,    46.000,    46.000,    46.000,    46.000,    46.000,
            42.000,    42.000,    42.000,    45.000,    45.000,    45.000,
            45.000,    43.000,    43.000,    43.000,    43.000,    37.000,
            37.000,    38.000,    38.000,    38.000,    38.000,    42.000,
            42.000,    42.000,    42.000,    42.000,    42.000,    47.000,
            47.000,    47.000,    47.000,    47.000,    47.000,    47.000,
            47.000,    47.000,    47.000,    47.000,    40.000,    40.000,
            40.000,    40.000,    45.000,    45.000,    45.000,    45.000,
            49.000,    49.000,    49.000,    46.000,    46.000,    46.000,
            41.000,    41.000,    41.000,    44.000,    44.000,    44.000,
            45.000,    45.000,    45.000,    45.000,    47.000,    47.000,
            47.000,    47.000,    46.000,    46.000,    46.000,    45.000,
            45.000,    45.000,    43.000,    43.000,    43.000],
        [  445.050,   445.050,   445.050,   445.050,   444.060,   444.060,
           444.060,   444.060,   450.870,   450.870,   450.870,   450.870,
           471.470,   471.470,   471.470,   462.380,   462.380,   462.380,
           462.380,   429.330,   429.330,   429.330,   429.330,   365.450,
           365.450,   408.530,   408.530,   408.530,   408.530,   447.580,
           447.580,   447.580,   490.510,   490.510,   490.510,   552.210,
           552.210,   552.210,   552.210,   470.790,   470.790,   470.790,
           470.790,   523.370,   523.370,   523.370,   450.070,   450.070,
           450.070,   450.070,   615.560,   615.560,   615.560,   615.560,
           540.690,   540.690,   540.690,   458.930,   458.930,   458.930,
           450.830,   450.830,   450.830,   444.030,   444.030,   444.030,
           442.340,   442.340,   442.340,   442.340,   553.940,   553.940,
           553.940,   553.940,   454.920,   454.920,   454.920,   448.660,
           448.660,   448.660,   461.020,   461.020,   461.020],
        [    0.123,     0.123,     0.123,     0.123,     0.130,     0.130,
             0.130,     0.130,     0.117,     0.117,     0.117,     0.117,
             0.092,     0.092,     0.092,     0.096,     0.096,     0.096,
             0.096,     0.161,     0.161,     0.161,     0.161,     0.143,
             0.143,     0.130,     0.130,     0.130,     0.130,     0.116,
             0.116,     0.116,     0.147,     0.147,     0.147,     0.175,
             0.175,     0.175,     0.175,     0.181,     0.181,     0.181,
             0.181,     0.167,     0.167,     0.167,     0.122,     0.122,
             0.122,     0.122,     0.196,     0.196,     0.196,     0.196,
             0.148,     0.148,     0.148,     0.132,     0.132,     0.132,
             0.102,     0.102,     0.102,     0.098,     0.098,     0.098,
             0.103,     0.103,     0.103,     0.103,     0.162,     0.162,
             0.162,     0.162,     0.140,     0.140,     0.140,     0.110,
             0.110,     0.110,     0.119,     0.119,     0.119]])

seen_table = torch.tensor([[    0.000,     0.000,     1.000,     1.000,     3.000,     1.000,
             2.000,     0.000,     1.000,     2.000,     3.000,     0.000,
             1.000,     0.000,     1.000,     0.000,     1.000,     0.000,
             0.000,     1.000,     0.000,     1.000,     3.000,     0.000,
             1.000,     3.000,     0.000,     1.000,     3.000,     0.000,
             1.000,     3.000],
        [  193.000,   212.000,   212.000,   244.000,   244.000,   151.000,
           148.000,   178.000,   178.000,   178.000,   178.000,   177.000,
           177.000,   183.000,   183.000,   202.000,   202.000,   216.000,
           197.000,   197.000,   227.000,   227.000,   227.000,   230.000,
           230.000,   230.000,   183.000,   183.000,   183.000,   216.000,
           216.000,   216.000],
        [  284.000,   289.000,   289.000,   309.000,   309.000,   230.000,
           210.000,   226.000,   226.000,   226.000,   226.000,   299.000,
           299.000,   252.000,   252.000,   280.000,   280.000,   335.000,
           306.000,   306.000,   311.000,   311.000,   311.000,   305.000,
           305.000,   305.000,   299.000,   299.000,   299.000,   307.000,
           307.000,   307.000],
        [    5.750,     4.160,     4.160,     3.170,     3.170,     3.520,
             3.490,     2.130,     2.130,     2.130,     2.130,     4.790,
             4.790,     2.220,     2.220,     4.210,     4.210,     6.900,
            10.010,    10.010,     6.310,     6.310,     6.310,     4.670,
             4.670,     4.670,    10.840,    10.840,    10.840,     7.240,
             7.240,     7.240],
        [   46.000,    46.000,    46.000,    45.000,    45.000,    43.000,
            37.000,    38.000,    38.000,    38.000,    38.000,    47.000,
            47.000,    47.000,    47.000,    40.000,    40.000,    49.000,
            46.000,    46.000,    41.000,    41.000,    41.000,    44.000,
            44.000,    44.000,    46.000,    46.000,    46.000,    45.000,
            45.000,    45.000],
        [  444.060,   450.870,   450.870,   462.380,   462.380,   429.330,
           365.450,   408.530,   408.530,   408.530,   408.530,   552.210,
           552.210,   523.370,   523.370,   450.070,   450.070,   540.690,
           458.930,   458.930,   450.830,   450.830,   450.830,   444.030,
           444.030,   444.030,   454.920,   454.920,   454.920,   448.660,
           448.660,   448.660],
        [    0.130,     0.117,     0.117,     0.096,     0.096,     0.161,
             0.143,     0.130,     0.130,     0.130,     0.130,     0.175,
             0.175,     0.167,     0.167,     0.122,     0.122,     0.148,
             0.132,     0.132,     0.102,     0.102,     0.102,     0.098,
             0.098,     0.098,     0.140,     0.140,     0.140,     0.110,
             0.110,     0.110]])

unseen_table = torch.tensor([[    0.000,     1.000,     2.000,     3.000,     1.000,     0.000,
             1.000,     2.000,     3.000,     2.000,     0.000,     1.000,
             2.000,     3.000],
        [  223.000,   223.000,   223.000,   223.000,   148.000,   198.000,
           198.000,   198.000,   198.000,   216.000,   208.000,   208.000,
           208.000,   208.000],
        [  302.000,   302.000,   302.000,   302.000,   210.000,   258.000,
           258.000,   258.000,   258.000,   335.000,   311.000,   311.000,
           311.000,   311.000],
        [    4.280,     4.280,     4.280,     4.280,     3.490,     2.750,
             2.750,     2.750,     2.750,     6.900,     6.370,     6.370,
             6.370,     6.370],
        [   43.000,    43.000,    43.000,    43.000,    37.000,    42.000,
            42.000,    42.000,    42.000,    49.000,    38.000,    38.000,
            38.000,    38.000],
        [  466.380,   466.380,   466.380,   466.380,   365.450,   428.760,
           428.760,   428.760,   428.760,   540.690,   485.180,   485.180,
           485.180,   485.180],
        [    0.115,     0.115,     0.115,     0.115,     0.143,     0.120,
             0.120,     0.120,     0.120,     0.148,     0.125,     0.125,
             0.125,     0.125]])

# =====================================================================================
# ported from train.py -- data pipeline (transforms, datasets, loaders). The
# whole_transform pipeline order (Grayscale -> RandomCrop(512) -> ColorJitter ->
# ToTensor -> Normalize) and the separate per-batch aug_transform are copied verbatim.
# =====================================================================================

whole_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomCrop(512),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomCrop(512),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

train_dataset = datasets.ImageFolder(whole_dir, transform=whole_transform)
seen_test_dataset = datasets.ImageFolder(seen_test_dir, transform=test_transform)
unseen_test_dataset = datasets.ImageFolder(unseen_test_dir, transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
seen_test_loader = torch.utils.data.DataLoader(dataset=seen_test_dataset, batch_size=39, shuffle=True)
unseen_test_loader = torch.utils.data.DataLoader(dataset=unseen_test_dataset, batch_size=33, shuffle=True)

l = len(train_loader)


# =====================================================================================
# ported from microstructure-to-properties.ipynb -- checkpoint I/O, kept in the exact
# same dict shape ("model_state"/"ema_model_state"/"model_optimizer") so any checkpoint
# produced here stays loadable by the original notebook unmodified.
# =====================================================================================

def save_model(address):
    checkpoint = {"model_state": model.state_dict(),
                  "ema_model_state": ema_model.state_dict(),
                  "model_optimizer": optimizer.state_dict()}
    torch.save(checkpoint, address)


def load_model(address):
    checkpoint = torch.load(address)
    model.load_state_dict(checkpoint["model_state"])
    ema_model.load_state_dict(checkpoint["ema_model_state"])
    optimizer.load_state_dict(checkpoint["model_optimizer"])


# ported from microstructure-to-properties.ipynb / train.py -- class label unpacking
def class_maker(batch_size, labels, tab):
    Mag = torch.zeros([batch_size])
    YS = torch.zeros([batch_size])
    UTS = torch.zeros([batch_size])
    EL = torch.zeros([batch_size])
    E = torch.zeros([batch_size])
    K = torch.zeros([batch_size])
    N = torch.zeros([batch_size])

    for i in range(batch_size):
        Mag[i] = tab[0, labels[i]]
        YS[i] = tab[1, labels[i]]
        UTS[i] = tab[2, labels[i]]
        EL[i] = tab[3, labels[i]]
        E[i] = tab[4, labels[i]]
        K[i] = tab[5, labels[i]]
        N[i] = tab[6, labels[i]]

    return Mag, YS, UTS, EL, E, K, N


# ported from train.py -- label-noise augmentation applied to YS/UTS/EL/E/K/N targets
def noise(real, range_value):
    for i in range(len(real)):
        random_value = torch.randint(-10, 10, (1,))
        if random_value in range(-1, 1):
            random_noise = torch.randn(1)
            real[i] = real[i] + ((0.3 * range_value) ** 0.5) * random_noise.item()
    return real


# ported from microstructure-to-properties.ipynb -- test() evaluation function
def test(test_data, table):
    test_images, test_labels = next(iter(test_data))
    test_images = aug_transform(test_images)
    test_images = test_images.to(device)
    t_mag, t_ys, t_uts, t_el, t_e, t_k, t_n = class_maker(
        batch_size=test_labels.size(0), labels=test_labels, tab=table)

    t_mag = t_mag.long().to(device)
    t_ys = t_ys.float().to(device)
    t_uts = t_uts.float().to(device)
    t_el = t_el.float().to(device)
    t_e = t_e.float().to(device)
    t_k = t_k.float().to(device)
    t_n = t_n.float().to(device)
    test_predictions = ema_model(test_images, t_mag)
    test_ys_error = mse(t_ys, test_predictions[:, 0])
    test_uts_error = mse(t_uts, test_predictions[:, 1])
    test_el_error = mse(t_el, test_predictions[:, 2])
    test_e_error = mse(t_e, test_predictions[:, 3])
    test_k_error = mse(t_k, test_predictions[:, 4])
    test_n_error = mse(t_n, test_predictions[:, 5])
    test_total_error = (test_ys_error + test_uts_error + test_el_error
                         + test_e_error + test_k_error + test_n_error)

    return (test_ys_error, test_uts_error, test_el_error, test_e_error,
            test_k_error, test_n_error, test_total_error)


# ported from microstructure-to-properties.ipynb / train.py -- weight init helpers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('Norm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nnn = m.in_features
            y = 1.0 / np.sqrt(nnn)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


# ported from microstructure-to-properties.ipynb -- EMA class
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=0):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        if self.step == step_start_ema:
            print('EMA started!')
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# =====================================================================================
# ported from microstructure-to-properties.ipynb -- model architecture, copied
# verbatim (SelfAttention, DoubleConv, Down, Dense, Net_conditional).
# =====================================================================================

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()
        self.imsize = imsize
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, (out_channels - 1))
        )

        self.Mag_label = nn.Sequential(
            nn.Embedding(magnifications, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))

    def forward(self, x, Mag):
        x = self.maxpool_conv(x)
        magemb = self.Mag_label(Mag).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        x = torch.cat((x, magemb), dim=1)
        return x


class Dense(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(mid_channels, 6),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class Net_conditional(nn.Module):
    def __init__(self, c_in=1, num_classes=None):
        super().__init__()
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16, 32, 256)
        self.sa1 = SelfAttention(32, 256)
        self.down2 = Down(32, 64, 128)
        self.sa2 = SelfAttention(64, 128)
        self.down3 = Down(64, 128, 64)
        self.sa3 = SelfAttention(128, 64)
        self.down4 = Down(128, 256, 32)
        self.sa4 = SelfAttention(256, 32)
        self.down5 = Down(256, 512, 16)
        self.sa5 = SelfAttention(512, 16)
        self.down6 = Down(512, 512, 8)
        self.sa6 = SelfAttention(512, 8)

        self.bot1 = DoubleConv(512, 256)
        self.bot2 = DoubleConv(256, 128)
        self.bot3 = DoubleConv(128, 64)

        self.linear = Dense(64 * 8 * 8, 64 * 8 * 8 * 2)

    def forward(self, x, Mag):
        x = self.inc(x)
        x = self.down1(x, Mag)
        x = self.down2(x, Mag)
        x = self.down3(x, Mag)
        x = self.down4(x, Mag)
        x = self.sa4(x)
        x = self.down5(x, Mag)
        x = self.sa5(x)
        x = self.down6(x, Mag)
        x = self.sa6(x)

        x = self.bot1(x)
        x = self.bot2(x)
        x = self.bot3(x)

        x = self.linear(x)
        return x


# =====================================================================================
# ported from train.py -- model/optimizer/EMA setup, with one change (below):
# =====================================================================================

model = Net_conditional(num_classes=num_classes).to(device)
model.apply(initialize_weights)  # ported from train.py -- explicit, reproducible init
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

# new -- added to convert continuation run into from-scratch training.
# train.py had `load_dir = '.../Mic-Mech-Over-800.pth.tar'; load_model(load_dir)` here,
# which restored an already-trained checkpoint right before the training loop -- that
# turns the run into fine-tuning, not training from scratch. That call is intentionally
# NOT made. `resume_dir` above is left as a manual opt-in only, for restarting an
# interrupted from-scratch run on Kaggle's ephemeral sessions.
if resume_dir is not None:
    print(f'Resuming from checkpoint: {resume_dir}')
    load_model(resume_dir)


# =====================================================================================
# new -- early stopping bookkeeping (patience-based) on top of train.py's existing
# min_loss / "save best" pattern, since ~800 epochs from scratch is long enough that
# it should stop before wasting the remainder of a free Kaggle GPU session.
# =====================================================================================

epochs_without_improvement = 0

seen_test_results = torch.zeros(n_epoch, 7)
unseen_test_results = torch.zeros(n_epoch, 7)
train_loss_epoch = []

# ported from train.py -- training loop (loss composition, EMA step, checkpointing)
for epoch in range(1, n_epoch + 1):
    model.train()
    loss_epoch = 0
    loss_ys_total = 0
    loss_uts_total = 0
    loss_el_total = 0
    loss_e_total = 0
    loss_k_total = 0
    loss_n_total = 0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = aug_transform(images)
        images = images.to(device)
        mag, ys, uts, el, e, k, n = class_maker(batch_size=labels.size(0), labels=labels, tab=class_table)
        ys = noise(ys, ys_range)
        uts = noise(uts, uts_range)
        el = noise(el, el_range)
        e = noise(e, e_range)
        k = noise(k, k_range)
        n = noise(n, n_range)

        mag = mag.long().to(device)
        ys = ys.float().to(device)
        uts = uts.float().to(device)
        el = el.float().to(device)
        e = e.float().to(device)
        k = k.float().to(device)
        n = n.float().to(device)

        labels = labels.long().to(device)
        predictions = model(images, mag)

        loss_ys = mse(ys.float(), predictions[:, 0].float())
        loss_uts = mse(uts.float(), predictions[:, 1].float())
        loss_el = mse(el.float(), predictions[:, 2].float())
        loss_e = mse(e.float(), predictions[:, 3].float())
        loss_k = mse(k.float(), predictions[:, 4].float())
        loss_n = mse(n.float(), predictions[:, 5].float())

        loss_ys_total += loss_ys.item()
        loss_uts_total += loss_uts.item()
        loss_el_total += loss_el.item()
        loss_e_total += loss_e.item()
        loss_k_total += loss_k.item()
        loss_n_total += loss_n.item()

        # ported from train.py -- deliberate scale-balancing weights across outputs
        # with very different numeric ranges (not present in the paper text)
        loss = 0.223 * loss_ys + 0.189 * loss_uts + 17.18 * loss_el + 12.32 * loss_e + 0.094 * loss_k + 675.68 * loss_n * 10

        loss.backward()

        optimizer.step()
        ema.step_ema(ema_model, model)
        loss_epoch += loss.item()

    train_loss_epoch.append(loss_epoch / len(train_loader))
    print('Training')
    print('Epoch: [%d/%d]: Loss: %.3f' % (epoch, n_epoch, loss_epoch / len(train_loader)))
    print(f'YS: {loss_ys_total/l} \t UTS: {loss_uts_total/l} \t el: {loss_el_total/l} \t E: {loss_e_total/l} \t k: {loss_k_total/l} \t n: {loss_n_total/l}')

    model.eval()
    with torch.no_grad():
        s_test_ys_error, s_test_uts_error, s_test_el_error, s_test_e_error, s_test_k_error, s_test_n_error, s_test_total_error = test(seen_test_loader, seen_table)
        u_test_ys_error, u_test_uts_error, u_test_el_error, u_test_e_error, u_test_k_error, u_test_n_error, u_test_total_error = test(unseen_test_loader, unseen_table)

        seen_test_results[epoch - 1, 0] = s_test_ys_error.item()
        seen_test_results[epoch - 1, 1] = s_test_uts_error.item()
        seen_test_results[epoch - 1, 2] = s_test_el_error.item()
        seen_test_results[epoch - 1, 3] = s_test_e_error.item()
        seen_test_results[epoch - 1, 4] = s_test_k_error.item()
        seen_test_results[epoch - 1, 5] = s_test_n_error.item()
        seen_test_results[epoch - 1, 6] = s_test_total_error.item()

        unseen_test_results[epoch - 1, 0] = u_test_ys_error.item()
        unseen_test_results[epoch - 1, 1] = u_test_uts_error.item()
        unseen_test_results[epoch - 1, 2] = u_test_el_error.item()
        unseen_test_results[epoch - 1, 3] = u_test_e_error.item()
        unseen_test_results[epoch - 1, 4] = u_test_k_error.item()
        unseen_test_results[epoch - 1, 5] = u_test_n_error.item()
        unseen_test_results[epoch - 1, 6] = u_test_total_error.item()

        print('Testing')
        print(f'Seen Loss: {s_test_total_error} \t Unseen Loss: {u_test_total_error}')

        combined_val_loss = (s_test_total_error + u_test_total_error).item()

        # ported from train.py -- "save best" pattern (min_loss comparison / save_model)
        if combined_val_loss < min_loss - min_delta:
            min_loss = combined_val_loss
            epochs_without_improvement = 0
            model_save_dir = os.path.join(save_dir, f'Mic-Mech-Scratch-Best{n_epoch}.pth.tar')
            save_model(model_save_dir)
            print('model saved!')
        else:
            # new -- early stopping on combined seen+unseen validation loss
            epochs_without_improvement += 1
            print(f'No improvement for {epochs_without_improvement}/{patience} epochs')
        print('----------------------')

    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch} (patience={patience} reached).')
        break

completed_epochs = epoch  # new -- actual epochs run, may be < n_epoch due to early stopping

model_save_dir = os.path.join(save_dir, f'Mic-Mech-Scratch-Final{n_epoch}.pth.tar')
save_model(model_save_dir)

# ported from train.py -- CSV export, adjusted (new) to use completed_epochs instead of
# a hardcoded epoch count, so rows match actual training length under early stopping
filename = 'Train' + str(trial)
filenamecsv = os.path.join(save_dir, filename + '.csv')
with open(filenamecsv, 'w', newline='') as csvfile:
    fieldnames = ['Epoch', 'Loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(1, completed_epochs + 1):
        writer.writerow({'Epoch': i, 'Loss': train_loss_epoch[i - 1]})

filename = 'Seen-Test' + str(trial)
filenamecsv = os.path.join(save_dir, filename + '.csv')
with open(filenamecsv, 'w', newline='') as csvfile:
    fieldnames = ['Epoch', 'YS Loss', 'UTS Loss', 'EL Loss', 'E Loss', 'k Loss', 'n Loss', 'Total Loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(1, completed_epochs + 1):
        writer.writerow({'Epoch': i, 'YS Loss': seen_test_results[i - 1, 0].item(), 'UTS Loss': seen_test_results[i - 1, 1].item(), 'EL Loss': seen_test_results[i - 1, 2].item(), 'E Loss': seen_test_results[i - 1, 3].item(), 'k Loss': seen_test_results[i - 1, 4].item(), 'n Loss': seen_test_results[i - 1, 5].item(), 'Total Loss': seen_test_results[i - 1, 6].item()})

filename = 'Unseen-Test' + str(trial)
filenamecsv = os.path.join(save_dir, filename + '.csv')
with open(filenamecsv, 'w', newline='') as csvfile:
    fieldnames = ['Epoch', 'YS Loss', 'UTS Loss', 'EL Loss', 'E Loss', 'k Loss', 'n Loss', 'Total Loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(1, completed_epochs + 1):
        writer.writerow({'Epoch': i, 'YS Loss': unseen_test_results[i - 1, 0].item(), 'UTS Loss': unseen_test_results[i - 1, 1].item(), 'EL Loss': unseen_test_results[i - 1, 2].item(), 'E Loss': unseen_test_results[i - 1, 3].item(), 'k Loss': unseen_test_results[i - 1, 4].item(), 'n Loss': unseen_test_results[i - 1, 5].item(), 'Total Loss': unseen_test_results[i - 1, 6].item()})

if device.type == 'cuda':
    print(torch.cuda.memory_summary(device=device))
