[README.md](https://github.com/user-attachments/files/34294504/README.md)
# Conser-vision: Wildlife Image Classification

A PyTorch computer-vision project for the [DrivenData Conser-vision Practice Area](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/).

The goal is to classify camera-trap images into eight mutually exclusive classes:

`antelope_duiker`, `bird`, `blank`, `civet_genet`, `hog`, `leopard`, `monkey_prosimian`, `rodent`.

> **Scope note:** the initial university concept document focused on wild birds. This competition contains seven wildlife groups plus a `blank` class, so this repository follows the official competition definition. The model is an educational prototype, not a guaranteed species identification system.

## Approach

- **Framework:** PyTorch + torchvision
- **Baseline:** ImageNet-pretrained ResNet18 (transfer learning)
- **Validation:** hold out camera-trap `site` groups to reduce location leakage
- **Metric:** multiclass log loss
- **Output:** class probabilities in the exact DrivenData submission format

The validation split is grouped by site because the competition test set contains sites not represented in the training set. A random image-level split can give an overly optimistic estimate.

## Repository structure

```text
conservision-wildlife-classifier/
├── data/
│   └── raw/                 # competition data (not committed)
├── models/                  # trained checkpoints (not committed)
├── submissions/             # generated CSV files (not committed)
├── src/
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_submission.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Get the data

1. Join the competition and accept its data-use rules on DrivenData.
2. Download the competition archive from its **Data** tab.
3. Extract the files so the directory looks like this:

```text
data/raw/
├── train_features.csv
├── train_labels.csv
├── test_features.csv
├── submission_format.csv       # if included in the downloaded archive
├── train_features/
│   ├── ZJ000000.jpg
│   └── ...
└── test_features/
    ├── ZJ016556.jpg
    └── ...
```

Competition data is intentionally excluded from this Git repository. Do not upload the images or private competition files to GitHub.

## 2. Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train

From the repository root:

```bash
python -m src.train --data-dir data/raw --epochs 10 --batch-size 32
```

For a quick smoke test:

```bash
python -m src.train --data-dir data/raw --epochs 1 --batch-size 8 --device cpu
```

The best checkpoint and training metadata are written to `models/`.

The first run downloads the publicly available ImageNet weights for ResNet18. If you do not have internet access, change the model initialization in `src/train.py` to `weights=None`.

## 4. Generate predictions

After training:

```bash
python -m src.predict --data-dir data/raw --checkpoint models/best_model.pt
```

The script creates `submissions/submission.csv`. It writes one probability per class, preserves the test ID order and checks that each row sums to approximately 1.

Upload this CSV on the DrivenData **Submit** page.

## 5. Tests

```bash
pytest -q
```

## Current limitations / next steps

This is a first reproducible baseline, not a claim of a high leaderboard score. Useful next experiments:

1. Compare ResNet18 with EfficientNet or ConvNeXt.
2. Tune image size, learning rate and augmentation.
3. Use grouped cross-validation by `site`.
4. Inspect per-class log loss and the confusion matrix.
5. Test test-time augmentation and probability calibration.

Record all validation results before selecting a final model. Never use the competition test labels (they are not provided).

## Data and competition

Dataset provided through DrivenData in collaboration with the Wild Chimpanzee Foundation and the Max Planck Institute for Evolutionary Anthropology. Download the data directly from the competition page and follow its terms of use.

Competition: https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/
