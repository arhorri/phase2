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

All directories are set at the top of `train_from_scratch.py`, overridable via
environment variables:

| Variable | Env var | Default |
|---|---|---|
| Training data | `WHOLE_DIR` | `data/Train-Oversampled` |
| Seen test data | `SEEN_TEST_DIR` | `data/Test-Seen` |
| Unseen test data | `UNSEEN_TEST_DIR` | `data/Test-Unseen` |
| Checkpoint/CSV output | `SAVE_DIR` | `checkpoints/` |
| Optional resume checkpoint | `RESUME_DIR` | unset (no resume — true from-scratch run) |

Each directory should be structured the same way as an `ImageFolder` dataset (one
subfolder per class, matching `label_dict`/`seen_label`/`unseen_label` in the script).

Datasets and checkpoints are **not** committed to this repo (see `.gitignore`).
Attach the training/test image folders and any checkpoint you want to resume from as a
Kaggle Dataset input, then point the env vars above at the mounted
`/kaggle/input/<dataset-name>/...` paths.

## Running on Kaggle

In a fresh Kaggle notebook (GPU enabled, P100 or T4):

```bash
!git clone https://github.com/arhorri/phase2.git
%cd phase2
!pip install -r requirements.txt
```

Then, after attaching your dataset(s) as Kaggle Dataset inputs and setting the
directory env vars to match their mounted paths:

```python
import os
os.environ["WHOLE_DIR"] = "/kaggle/input/<dataset>/Train-Oversampled"
os.environ["SEEN_TEST_DIR"] = "/kaggle/input/<dataset>/Test-Seen"
os.environ["UNSEEN_TEST_DIR"] = "/kaggle/input/<dataset>/Test-Unseen"
os.environ["SAVE_DIR"] = "/kaggle/working/checkpoints"

!python train_from_scratch.py
```

To run interactively instead of as a script, open
[train_from_scratch.ipynb](train_from_scratch.ipynb) in the Kaggle notebook editor
(File > Import Notebook, or copy it into a new Kaggle notebook cell-by-cell). Its
second cell ("Kaggle setup") sets the same environment variables — edit it to match
your attached dataset slugs, then run all cells in order.

### Troubleshooting: `FileNotFoundError: data/Train-Oversampled`

This means `WHOLE_DIR`/`SEEN_TEST_DIR`/`UNSEEN_TEST_DIR` were never pointed at your
Kaggle dataset, so the script fell back to its local-run defaults. Two things to check:

1. **You need the image dataset attached, not just a checkpoint dataset.** A dataset
   like `mic-mech3-checkpoint` (containing only `.pth.tar` files and CSVs) does not
   contain the `Train-Oversampled`/`Test-Seen`/`Test-Unseen` image folders — you need
   a separate dataset with those `ImageFolder`-structured directories attached under
   "Datasets" in the notebook sidebar.
2. **Set the env vars before the data-pipeline cell runs**, using the actual mounted
   path — Kaggle mounts each attached dataset at `/kaggle/input/<dataset-slug>/...`.
   Check the exact slug and folder names in the sidebar and set, e.g.:
   ```python
   import os
   os.environ["WHOLE_DIR"] = "/kaggle/input/<your-image-dataset>/.../Train-Oversampled"
   os.environ["SEEN_TEST_DIR"] = "/kaggle/input/<your-image-dataset>/.../Test-Seen"
   os.environ["UNSEEN_TEST_DIR"] = "/kaggle/input/<your-image-dataset>/.../Test-Unseen"
   os.environ["SAVE_DIR"] = "/kaggle/working/checkpoints"
   ```
   (`train_from_scratch.ipynb` has this as its "Kaggle setup" cell already — just edit
   the paths to match your dataset slug.)

Checkpoints are saved in the same `{"model_state", "ema_model_state",
"model_optimizer"}` format used by `train.py` and
`microstructure-to-properties.ipynb`, so a checkpoint produced by this script can be
loaded directly by the original inference notebook without modification.
