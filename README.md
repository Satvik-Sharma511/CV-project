# 🛰️ Satellite Image Land Classification

A deep learning web application that classifies satellite images into 10 land-use categories using transfer learning (MobileNetV2) and a Streamlit interface — with built-in image enhancement and feature extraction tools.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Environment Setup](#environment-setup)
6. [Dependency Installation](#dependency-installation)
7. [Dataset Configuration](#dataset-configuration)
8. [Training the Model](#training-the-model)
9. [Running the Application](#running-the-application)
10. [Class Labels](#class-labels)
11. [Configuration Reference](#configuration-reference)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project uses the **EuroSAT** dataset and a fine-tuned **MobileNetV2** backbone to classify satellite imagery into 10 distinct land-use classes. The application exposes the full ML pipeline — from image enhancement through feature extraction to classification — via a Streamlit web dashboard.

---

## Features

- **Image Enhancement** — CLAHE, denoising, brightness/contrast, gamma correction, sharpening
- **Feature Extraction** — Canny/Sobel/Laplacian edges, ORB/SIFT/BRISK keypoints, HOG descriptors, Gabor textures, contour detection, threshold segmentation
- **Land Classification** — MobileNetV2 transfer learning, top-K predictions, confidence scores, probability chart
- **Downloadable Outputs** — Enhanced image and edge maps available for download
- **In-App Training** — Trigger model training directly from the Streamlit sidebar

---

## Project Structure

```
satellite_image_land_classification/
├── app.py                  # Streamlit entry point
├── train_model.py          # Model training pipeline
├── run_app.bat             # Windows launcher script
├── data/
│   └── EuroSAT/            # Dataset (one subfolder per class)
│       ├── AnnualCrop/
│       ├── Forest/
│       └── ...
├── model/
│   ├── land_classifier_model.keras   # Saved trained model
│   └── class_labels.json             # Class label index
└── utils/
    ├── classification.py       # Inference utilities
    ├── feature_extraction.py   # CV feature extractors
    └── image_enhancement.py    # Preprocessing pipeline
```

---

## Requirements

| Requirement | Recommended Version |
|-------------|-------------------|
| Python      | **3.10.x**        |
| pip         | ≥ 23.0            |
| OS          | Windows 10/11, macOS, or Linux |
| RAM         | ≥ 8 GB (16 GB recommended for training) |
| Disk Space  | ≥ 5 GB (dataset + model) |

---

## Environment Setup

### Option A — Python venv (Recommended)

```bash
# 1. Clone or download the project
cd C:\Users\LENOVO\Desktop\satellite_image_land_classification

# 2. Create a virtual environment using Python 3.10
"C:\Users\LENOVO\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv

# 3. Activate the environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### Option B — Conda Environment

```bash
conda create -n landclassify python=3.10 -y
conda activate landclassify
```

---

## Dependency Installation

With your virtual environment **activated**, install all required packages:

```bash
pip install --upgrade pip

pip install tensorflow==2.13.0
pip install streamlit
pip install opencv-python
pip install opencv-contrib-python     # required for SIFT and BRISK
pip install numpy
pip install pandas
pip install matplotlib
pip install Pillow
pip install scikit-image
```

> **Note:** If you have an NVIDIA GPU and want to enable GPU acceleration for training, install `tensorflow-gpu==2.13.0` and the matching CUDA/cuDNN drivers instead of plain `tensorflow`.

### Verify Installation

```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
python -c "import streamlit; print('Streamlit OK')"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

---

## Dataset Configuration

1. **Download the EuroSAT dataset** from [https://github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT) (RGB version).

2. **Extract** the ZIP so the folder structure looks like this:

```
data/
└── EuroSAT/
    ├── AnnualCrop/
    │   ├── AnnualCrop_00001.jpg
    │   └── ...
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```

3. **Update the dataset path** in `train_model.py` if your data lives elsewhere:

```python
# Line ~20 in train_model.py
DATA_DIR = r"C:\Users\LENOVO\Desktop\satellite_image_land_classification\data\EuroSAT"
```

Change this to your actual path, e.g.:

```python
DATA_DIR = r"C:\Users\YourName\Downloads\EuroSAT"
```

---

## Training the Model

> **Skip this step** if you already have `model/land_classifier_model.keras` and `model/class_labels.json`.

### From the command line

```bash
# Make sure your venv is activated first
python train_model.py
```

Training will:
1. Validate the dataset structure
2. Load and augment training data
3. Build MobileNetV2 with a custom classification head
4. Train for up to 10 epochs (early stopping applies)
5. Save the best model to `model/land_classifier_model.keras`
6. Save class labels to `model/class_labels.json`
7. Display training accuracy/loss plots

### Tuning for low-spec hardware

Open `train_model.py` and reduce these values:

```python
IMAGE_SIZE = (96, 96)   # default: (128, 128)
BATCH_SIZE = 16         # default: 32
EPOCHS = 5              # default: 10
```

### From inside the app

You can also trigger training from the **Streamlit sidebar** → *Train Model* button (requires dataset to be configured first).

---

## Running the Application

### Method 1 — Windows Batch File (Easiest)

Double-click **`run_app.bat`** in the project folder. It uses the Python 3.10 interpreter directly and launches Streamlit automatically.

### Method 2 — Command Line

```bash
# Activate your environment
venv\Scripts\activate         # Windows
source venv/bin/activate      # macOS/Linux

# Launch the app
streamlit run app.py
```

### Method 3 — Specifying a Port

```bash
streamlit run app.py --server.port 8502
```

After launching, the app opens automatically in your browser at:

```
http://localhost:8501
```

---

## Using the App

1. **Upload** a satellite image (JPG or PNG) using the file uploader.
2. **Adjust** enhancement settings in the left sidebar (CLAHE, denoising, brightness, etc.).
3. Click **"Run Full Pipeline"** to process the image.
4. Review outputs in three sections:
   - **Image Enhancement** — before/after comparison, histogram
   - **Feature Extraction** — edges, keypoints, HOG, contours, Gabor responses
   - **Land Classification** — predicted class, confidence %, probability bar chart
5. **Download** enhanced image or edge maps using the download buttons.

---

## Class Labels

The model classifies images into these 10 EuroSAT categories:

| Index | Class Label             |
|-------|-------------------------|
| 0     | AnnualCrop              |
| 1     | Forest                  |
| 2     | HerbaceousVegetation    |
| 3     | Highway                 |
| 4     | Industrial              |
| 5     | Pasture                 |
| 6     | PermanentCrop           |
| 7     | Residential             |
| 8     | River                   |
| 9     | SeaLake                 |

---

## Configuration Reference

### Training (`train_model.py`)

| Variable           | Default         | Description                        |
|--------------------|-----------------|------------------------------------|
| `DATA_DIR`         | *(absolute path)* | Path to EuroSAT dataset folder   |
| `MODEL_DIR`        | `model`         | Output directory for saved model   |
| `IMAGE_SIZE`       | `(128, 128)`    | Input image dimensions             |
| `BATCH_SIZE`       | `32`            | Training batch size                |
| `EPOCHS`           | `10`            | Maximum training epochs            |
| `VALIDATION_SPLIT` | `0.2`           | Fraction of data used for validation |
| `LEARNING_RATE`    | `1e-3`          | Adam optimizer learning rate       |

### App (`app.py`)

All enhancement and classification settings are controlled interactively via the **Streamlit sidebar** sliders and checkboxes.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` error in app | Run `train_model.py` first, or place `land_classifier_model.keras` in the `model/` folder |
| SIFT not available | Install `opencv-contrib-python` (not just `opencv-python`) |
| `ModuleNotFoundError: streamlit` | Activate your venv before running; check `pip install streamlit` |
| Training runs out of memory | Reduce `BATCH_SIZE` to 16 and `IMAGE_SIZE` to `(96, 96)` |
| `FileNotFoundError` for dataset | Verify `DATA_DIR` path in `train_model.py` matches your actual folder |
| `TF_CPP_MIN_LOG_LEVEL` warnings | These are suppressed by default; if they appear, they are informational only |
| App crashes on image upload | Ensure the image is a valid JPG or PNG; try a different image |

---

## License

This project is for educational and research purposes. The EuroSAT dataset is subject to its own license — please refer to the [official EuroSAT repository](https://github.com/phelber/EuroSAT) for details.
