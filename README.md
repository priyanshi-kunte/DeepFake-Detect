# DeepFake-Detect

A deep learning pipeline for detecting deepfake videos/images using face-crop preprocessing and a CNN classifier (EfficientNetB0 backbone), with a Flask web app for interactive predictions.

## Overview

This project detects whether a face in an image/video frame is **real** or **fake** (deepfake-manipulated). It follows an end-to-end pipeline:

1. Extract frames from videos
2. Detect and crop faces from those frames
3. Organize cropped faces into a labeled real/fake dataset and split into train/val/test sets
4. Train a CNN classifier (EfficientNetB0 as feature extractor) on the face crops
5. Serve predictions through a simple Flask web interface

![Sample Dataset](img/sample_dataset.png)

## Project Structure

```
DeepFake-Detect/
│
├── 00-convert_video_to_image.py         # Extracts frames from raw videos
├── 01a-crop_faces_with_mtcnn.py         # Face detection & cropping using MTCNN
├── 01b-crop_faces_with_azure-vision-api.py  # Alternative face cropping via Azure Vision API
├── 02-prepare_fake_real_dataset.py      # Organizes crops into real/fake classes & splits train/val/test
├── 03-train_cnn.py                      # Trains the CNN (EfficientNetB0-based) classifier
├── app.py                               # Flask app for running predictions via a web UI
├── templates/
│   └── index.html                       # Front-end template for the Flask app
├── img/                                 # Screenshots used in this README
├── efficientnetb0_notop.h5              # Pre-trained EfficientNetB0 weights (no top layer) — not tracked in Git
├── requirements.txt                     # Python dependencies
├── .gitignore
└── LICENSE                              # MIT License
```

> **Note:** Folders like `data/`, `prepared_dataset/`, `split_dataset/`, `sample/`, and `train_sample_videos/` are excluded from version control (see `.gitignore`) since they contain large datasets. You'll need to regenerate or supply your own dataset locally to fully run the pipeline.

## How It Works

### 1. Frame Extraction
`00-convert_video_to_image.py` reads raw videos (e.g. from the [DFDC dataset](https://ai.meta.com/datasets/dfdc/)) and extracts individual frames as images.

### 2. Face Cropping
Faces are detected and cropped from each frame using one of two methods:
- `01a-crop_faces_with_mtcnn.py` — uses **MTCNN** for local face detection
- `01b-crop_faces_with_azure-vision-api.py` — uses **Azure Cognitive Services Vision API** for cloud-based face detection

### 3. Dataset Preparation
`02-prepare_fake_real_dataset.py` organizes cropped face images into `real/` and `fake/` classes and splits the data into `train/`, `val/`, and `test/` sets (default ratio: 80/10/10).

### 4. Model Training
`03-train_cnn.py` trains a CNN classifier using **EfficientNetB0** (pre-trained, top layer removed) as a feature extractor, fine-tuned on the prepared real/fake face dataset.

### 5. Web App
`app.py` launches a Flask server with a simple web interface (`templates/index.html`) where you can upload an image and get a real/fake prediction.

## Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/pratham-sachdeva7/DeepFake-Detect.git
cd DeepFake-Detect

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Dataset

This repo does **not** include the raw dataset or processed face crops (excluded via `.gitignore` due to size). To reproduce the pipeline:

1. Download a deepfake dataset such as the [DFDC (DeepFake Detection Challenge) dataset](https://ai.meta.com/datasets/dfdc/) and place raw videos in `train_sample_videos/`.
2. Run the pipeline scripts in order:
   ```bash
   python 00-convert_video_to_image.py
   python 01a-crop_faces_with_mtcnn.py
   python 02-prepare_fake_real_dataset.py
   python 03-train_cnn.py
   ```

### Running the Web App

```bash
python app.py
```

Then open your browser at `http://localhost:5000` (or the port shown in the console) to upload an image and get a prediction.

## Model

- **Backbone:** EfficientNetB0 (ImageNet pre-trained, top layer removed) — `efficientnetb0_notop.h5`
- **Task:** Binary classification (Real vs. Fake)
- **Input:** Cropped face images

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- MTCNN
- Flask
- HTML/CSS (Flask templates)

## License

This project is licensed under the [MIT License](LICENSE).

## Authors

- **Pratham Sachdeva**
- **Priyanshi Kunte**

## Acknowledgements

- Face cropping and CNN training approach adapted from open-source deepfake detection tutorials.
- Pre-trained EfficientNetB0 weights from Keras Applications.
