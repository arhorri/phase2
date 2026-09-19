# AZ80 Microstructure-to-Properties

Predicts the mechanical properties of a magnesium alloy (AZ80) directly from a
microscope photo of its internal structure — no physical mechanical test needed.

## 1. The problem this project solves

Normally, to know how strong or ductile a piece of metal is, you have to
machine a sample into a standard shape and physically pull it apart in a
tensile-testing machine. That's slow, destructive (you lose the sample), and
expensive to do for every batch of material.

This project instead trains a neural network to *look at* a microscope image
of the metal's microstructure (the pattern of grains and phases you'd see
under a microscope) and predict six mechanical properties directly:

- **YS** — Yield Strength (the stress at which the metal starts to permanently
  deform)
- **UTS** — Ultimate Tensile Strength (the maximum stress it can take before
  breaking)
- **EL** — Elongation (how much it stretches before breaking, as a %)
- **E** — Elastic Modulus (stiffness)
- **k** and **n** — the two constants of the material's stress–strain
  hardening curve (σ = k·εⁿ)

The idea: if the model works, you could estimate these properties from a
cheap, non-destructive micrograph instead of destroying a sample every time.

This is based on the published paper: E. Azqadan, A. Arami, H. Jahed,
*"From microstructure to mechanical properties: Image-based machine learning
prediction for AZ80 magnesium alloy,"* Journal of Magnesium and Alloys (2025),
https://doi.org/10.1016/j.jma.2025.07.019. Please cite the paper if you use
this model.

The model is evaluated two ways: **"seen"** conditions (new photos of a
processing condition it already saw examples of during training) and
**"unseen"** conditions (a processing condition it never saw at all) — the
second is the harder, more realistic test of whether it actually generalizes.

## 2. The dataset

The images live in [az80-microstructure-data/](az80-microstructure-data/).
Each image is a micrograph — a 1140×768 color photo taken through a
microscope at a specific magnification (500×, 1000×, 1500×, or 2000×).
Images are grouped into folders by *processing condition* (e.g.
`CM04-0500`, `PL13-1000`) — samples of AZ80 that were cast, forged, and
heat-treated differently from one another. **This repo does not contain a
legend decoding what the letters in those folder names mean** (only the
trailing number is known — it's the magnification); treat each folder as
just an opaque class ID.

Every image's true property values (YS/UTS/EL/E/k/n) are listed in
[dataset.csv](az80-microstructure-data/dataset.csv), one row per image.

What's actually in this repo:

| Folder | Classes | Images | Purpose |
|---|---|---|---|
| `Test-Seen` | 32 | 39 | held-out images of *known* processing conditions |
| `Test-Unseen` | 14 | 33 | images of processing conditions never trained on |
| `Train-Oversampled` | — | — | **not present** — see below |

**Which dataset to retrain on:** the paper's real training set,
`Train-Oversampled` (83 classes, artificially expanded by the authors using a
diffusion model), was never publicly released and isn't in this repo. So in
practice, retraining today has to use `az80-microstructure-data`
(`Test-Seen` + `Test-Unseen`) — the training scripts already do this
automatically (see [Section 5](#5-training-the-model-from-scratch)). Be aware
this is a *small* substitute dataset, and because there's no separate
training set, the same images end up used for both training and checking
progress — so it proves the code runs end-to-end, but the resulting model
isn't scientifically meaningful. If you ever obtain the real
`Train-Oversampled` folder, drop it into `az80-microstructure-data/` and the
scripts will automatically switch to using it instead.

## 3. Preparing the data for training

Before an image reaches the model, it goes through a few automatic steps:

1. **Convert to grayscale** — the model only looks at brightness patterns,
   not color.
2. **Random 512×512 crop** — a random square window is cut out of the
   (larger) source photo. Doing this randomly, instead of always cropping
   the same spot, helps the model learn from different parts of each image
   across training runs.
3. **Random brightness/contrast jitter** *(training only)* — slightly
   varies lighting/exposure so the model doesn't latch onto exact lighting
   conditions instead of the actual microstructure.
4. **Convert to a tensor and normalize** — standard step that rescales pixel
   values into a small numeric range that's easier for the network to learn
   from.
5. **Random flip (horizontal and/or vertical)** — applied to every image,
   train and test. Microstructure has no inherent "up" direction, so
   flipping it is a safe, free way to show the model more variety.

On top of that, during training only, the true property values themselves
get a small amount of random noise added about 10% of the time (sized to
each property's typical range). This is a mild regularizer — it stops the
model from treating the training labels as if they were perfectly precise
measurements.

One more thing worth knowing: the true property values aren't measured
per photo — every image inside the same class folder (e.g. every image in
`CM04-0500/`) is assigned the *same* fixed YS/UTS/EL/E/k/n values, looked up
by which folder it came from.

## 4. The model, step by step

The model is called `Net_conditional`. It's a convolutional network with a
self-attention block and a small regression head at the end.

**Inputs:**
- One grayscale image, 512×512 pixels
- The image's magnification level (500×, 1000×, 1500×, or 2000×), given
  as a side input

**What happens inside, step by step:**

1. An initial convolution block turns the raw image into a small stack of
   feature maps (16 channels).
2. The image is then shrunk down in **6 stages**. Each stage halves the
   image's width and height while increasing the number of feature
   channels (512→256→128→64→32→16→8 pixels; 16→32→64→128→256→512→512
   channels). At every one of these 6 stages, the model also mixes in the
   magnification value, so it always "knows" what zoom level it's looking
   at.
3. At the three most-shrunk stages (32×32, 16×16, and 8×8), the model
   applies **self-attention** — a mechanism that lets it directly relate
   distant regions of the image to each other, not just nearby pixels.
4. Three more convolution blocks refine these deep features further (a
   "bottleneck").
5. Finally, everything is flattened and passed through a couple of fully
   connected layers down to just 6 numbers.

**Output:** 6 numbers per image, always in this order:
**YS, UTS, EL, E, k, n** — the predicted mechanical properties.

## 5. Training the model from scratch

The easiest way is the notebook
[train_from_scratch.ipynb](train_from_scratch.ipynb) — it trains a brand-new,
randomly-initialized model (as opposed to [train.py](train.py), which
continues training an already-trained checkpoint).

### On Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) and
   open [train_from_scratch.ipynb](train_from_scratch.ipynb) — either upload
   it, or File → Open notebook → GitHub → `arhorri/phase2`.
2. Runtime → Change runtime type → select a GPU.
3. Runtime → Run all.
   - The first code cell automatically clones this repo (so the dataset in
     `az80-microstructure-data/` is available) and installs the required
     Python packages — no manual setup needed.
   - Because the real `Train-Oversampled` set isn't available, the notebook
     automatically builds a small training set from `Test-Seen` +
     `Test-Unseen` instead, and prints a warning explaining this.
   - Training then runs for up to 800 epochs, stopping early if it goes 50
     epochs with no improvement. Progress prints after every epoch, along
     with a live-updating loss plot (train/seen-val/unseen-val), and every
     10 epochs (configurable via `IMAGE_LOG_EVERY`) a grid of sample
     validation images with predicted-vs-true property values.
   - The trained model and CSV logs are saved under `checkpoints/`.

### On Kaggle

In a fresh Kaggle notebook (GPU enabled):

```bash
!git clone https://github.com/arhorri/phase2.git
%cd phase2
!pip install -r requirements.txt
!python train_from_scratch.py
```

Or open `train_from_scratch.ipynb` in the Kaggle notebook editor and run all
cells — same automatic behavior as Colab.

### Using a different dataset location, or your own data

If you want to point the scripts at data stored somewhere else, set these
before running (as environment variables, or `os.environ[...] = ...` in a
notebook cell before the Config cell runs):

| What it controls | Env var | Default |
|---|---|---|
| Training images | `WHOLE_DIR` | `az80-microstructure-data/Train-Oversampled` |
| "Seen" test images | `SEEN_TEST_DIR` | `az80-microstructure-data/Test-Seen` |
| "Unseen" test images | `UNSEEN_TEST_DIR` | `az80-microstructure-data/Test-Unseen` |
| Where checkpoints/logs are saved | `SAVE_DIR` | `checkpoints/` |
| A checkpoint to resume from (optional) | `RESUME_DIR` | unset |

A new dataset must be laid out the same way as `Test-Seen`/`Test-Unseen`:
one subfolder per class, images directly inside. One extra thing to know:
because the model's true property values are looked up per-class (see
Section 3), plugging in a genuinely different dataset also means providing a
matching table of YS/UTS/EL/E/k/n values for each new class — the code
doesn't read property values from anywhere else.

Checkpoints are saved in the same format used by the original project, so
a model trained here can be loaded directly by
[microstructure-to-properties.ipynb](microstructure-to-properties.ipynb)
without any changes.
