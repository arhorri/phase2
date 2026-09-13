# AZ80 Microstructure-to-Properties: From-Scratch Training

Trains `Net_conditional` (image -> YS/UTS/EL/E/k/n) from randomly initialized weights,
based on the model in
[microstructure-to-properties.ipynb](microstructure-to-properties.ipynb) and the
training procedure in [train.py](train.py) (the project's original training script).

`train.py` is kept unmodified as a reference: it is actually a **continuation run** —
it loads a checkpoint already trained for 800 epochs
(`Mic-Mech-Over-800.pth.tar`) and trains 200 more epochs on top of it.
[train_from_scratch.py](train_from_scratch.py) is the new script that trains from
scratch instead, and [train_from_scratch.ipynb](train_from_scratch.ipynb) is the same
script as a notebook (same cells, same code, split into sections with markdown
explanations) for running interactively on Kaggle.

## What changed vs. train.py

- The `load_model(load_dir)` call that restored the pretrained checkpoint before the
  training loop is removed. `model.apply(initialize_weights)` is kept, so
  initialization is still explicit and reproducible.
- `min_loss` starts at `+inf` instead of the `1848.41` value train.py inherited from
  its prior run.
- Early stopping was added on the combined seen+unseen validation loss.
- CSV export row counts follow the actual number of epochs run (relevant if early
  stopping triggers), instead of a hardcoded epoch count.
- All dataset/checkpoint directories are read from environment variables (or default
  to local relative paths) instead of hardcoded `/kaggle/input/...` paths.

Everything else — batch size, transforms, the label-noise `noise()` function, the
weighted loss composition, the class/property lookup tables, the EMA setup, the
checkpoint dict format, learning rate/optimizer/loss function — is ported verbatim
from train.py. Each block in `train_from_scratch.py` is commented as either
`ported from microstructure-to-properties.ipynb`, `ported from train.py`, or
`new -- added to convert continuation run into from-scratch training`.

## Hyperparameters: source of each value

| Value | Source | Notes |
|---|---|---|
| `batch_size = 4` | train.py | used as-is; kept configurable via the variable at the top |
| `learning_rate = 3e-5` | train.py | Adam optimizer |
| loss weights (0.223, 0.189, 17.18, 12.32, 0.094, 675.68×10) | train.py | not in the paper text |
| `whole_transform` / `aug_transform` pipelines | train.py | order and params kept exact |
| `ys_range`/`uts_range`/`el_range`/`e_range`/`k_range`/`n_range` | train.py | used by `noise()` |
| `n_epoch = 800` | **paper**, Section 3.2 ("training was completed after approximately 800 epochs") | train.py itself only ever ran 200-epoch continuation windows, so it has no from-scratch total epoch count to port |
| `patience = 50`, `min_delta = 0.0` | **new — our own default** | not specified anywhere in the paper, notebook, or train.py; the paper only says early stopping was used, not with what patience |
| `min_loss` starting value = `+inf` | **new** | train.py's `1848.41` was a value inherited from a prior checkpoint's validation loss, not a from-scratch baseline |

If you have a better source for the true early-stopping patience (e.g. from
correspondence with the authors), edit `patience` at the top of
`train_from_scratch.py` — it is not something this port could recover from the
provided files.

## Configuring paths

The dataset is committed directly in this repo under `az80-microstructure-data/`
(`ImageFolder`-structured: one subfolder per class, matching
`label_dict`/`seen_label`/`unseen_label` in the script), so the defaults in
`train_from_scratch.py`/`train_from_scratch.ipynb` work unchanged locally, in Colab,
or after `git clone` on Kaggle — no dataset attachment step required.

| Variable | Env var | Default |
|---|---|---|
| Training data | `WHOLE_DIR` | `az80-microstructure-data/Train-Oversampled` |
| Seen test data | `SEEN_TEST_DIR` | `az80-microstructure-data/Test-Seen` |
| Unseen test data | `UNSEEN_TEST_DIR` | `az80-microstructure-data/Test-Unseen` |
| Checkpoint/CSV output | `SAVE_DIR` | `checkpoints/` |
| Optional resume checkpoint | `RESUME_DIR` | unset (no resume — true from-scratch run) |

Override any of these as environment variables only if you keep the data elsewhere
(e.g. a separate Kaggle Dataset mounted under `/kaggle/input/...`).

**Current status:** `az80-microstructure-data/` only contains `Test-Seen` and
`Test-Unseen` so far — `Train-Oversampled` still needs to be added (83 classes, matching
`class_table` in the script) before the training loop can run. Both scripts check for
all three directories up front and raise a clear `FileNotFoundError` naming the missing
one, instead of the raw error from deep inside `ImageFolder`. Checkpoints and CSV
outputs are still gitignored (see `.gitignore`) — only the image data itself is tracked.

## Running on Kaggle

In a fresh Kaggle notebook (GPU enabled, P100 or T4):

```bash
!git clone https://github.com/arhorri/phase2.git
%cd phase2
!pip install -r requirements.txt
!python train_from_scratch.py
```

No dataset attachment or path editing is needed unless you want to override the
defaults (see the table above).

To run interactively instead of as a script, open
[train_from_scratch.ipynb](train_from_scratch.ipynb) in the Kaggle notebook editor
(File > Import Notebook, or copy it into a new Kaggle notebook cell-by-cell) and run
all cells in order.

Checkpoints are saved in the same `{"model_state", "ema_model_state",
"model_optimizer"}` format used by `train.py` and
`microstructure-to-properties.ipynb`, so a checkpoint produced by this script can be
loaded directly by the original inference notebook without modification.
