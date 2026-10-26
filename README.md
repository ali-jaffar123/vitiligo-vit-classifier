# Vitiligo Image Classification with Vision Transformer (ViT)

An image classification project that fine-tunes the Google/Meta Vision Transformer (`google/vit-base-patch16-224`) to tell vitiligo skin images apart from healthy skin images. The trained model is served through a FastAPI web application with an upload-and-predict frontend.

This project was completed as the Week 08 class assignment for the NIAI program.

## Disclaimer

This is a student machine learning project. It is not a medical device and must not be used for diagnosis. Please consult a qualified dermatologist for any skin concern.

## Overview

Vitiligo is a skin condition in which pigment is lost, producing white patches on different parts of the body. Recognising it from photos is difficult because the appearance of lesions changes with skin tone, lighting, and background. This project trains a ViT classifier on a public Kaggle dataset and evaluates it with standard classification metrics.

## Dataset

- Source: [Vitiligo Dataset on Kaggle](https://www.kaggle.com/datasets/zienabesam/vitiligo-dataset)
- Classes: `Healthy Skin` and `vitiligo`
- The dataset is not included in this repository because of its size. Download it from the link above.

Expected folder layout (used by `torchvision.datasets.ImageFolder`):

```
Vitiligo DS/
    Healthy Skin/
    vitiligo/
```

## Method

| Item | Setting |
| --- | --- |
| Base model | `google/vit-base-patch16-224` (pretrained on ImageNet) |
| Input size | 224 x 224, normalised with the ViT processor mean and std |
| Augmentation (train only) | Random horizontal flip, random rotation (10 degrees), brightness and contrast jitter |
| Split | 70 percent train, 15 percent validation, 15 percent test (stratified, seed 42) |
| Loss | Cross-entropy |
| Optimiser | AdamW, learning rate 2e-5 |
| Batch size | 16 |
| Epochs | 5 |
| Model selection | Weights from the epoch with the best validation accuracy |

## Evaluation

The model is evaluated on the held-out test set with:

- Accuracy
- Precision, recall, and F1-score per class
- Confusion matrix
- ROC curve and ROC-AUC
- Training and validation loss and accuracy curves

### Results

Fill in these values from your own run of the notebook.

| Metric | Value |
| --- | --- |
| Test accuracy | |
| Precision (vitiligo) | |
| Recall (vitiligo) | |
| F1-score (vitiligo) | |
| ROC-AUC | |

## Repository Structure

```
.
├── Vitiligo_ViT_Classifier.ipynb   Training, evaluation, and Colab web app cells
├── app/
│   ├── main.py                     FastAPI backend (local use)
│   └── static/
│       └── index.html              Frontend page
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Option 1: Run everything in Google Colab

1. Upload the dataset folder to Google Drive (for example `MyDrive/Vitiligo DS`).
2. Open `Vitiligo_ViT_Classifier.ipynb` in Google Colab and set the runtime to GPU.
3. Run the training and evaluation cells in order. The trained model is saved to `MyDrive/vitiligo_vit_model`.
4. Run the final web app cells. They start the FastAPI server and print a public `trycloudflare.com` link that opens the frontend in any browser. The link stays active while the Colab session is running.

### Option 2: Run the web app locally

The trained model weights are not stored in this repository. Download the `vitiligo_vit_model` folder from your Google Drive first.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

# place the downloaded model folder next to main.py, or set MODEL_PATH
cd app
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

To use a model folder in a different location:

```bash
# Linux / macOS
MODEL_PATH="/path/to/vitiligo_vit_model" uvicorn main:app --reload

# Windows (cmd)
set MODEL_PATH=C:\path\to\vitiligo_vit_model
uvicorn main:app --reload
```

## Web Application

The frontend lets a user:

- Upload an image with a button or by drag and drop
- See a preview of the selected image with its file name, size, and dimensions
- Run the model and view the predicted class with a confidence score
- See a probability bar for each class
- Receive a low-confidence warning when the model is unsure

### API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Serves the frontend page |
| GET | `/health` | Returns server status, device, and class names |
| POST | `/predict` | Accepts an image file and returns the prediction, confidence, and per-class probabilities |

Example request:

```bash
curl -X POST -F "file=@sample.jpg" http://127.0.0.1:8000/predict
```

## Limitations

- The model knows only two classes. It will return a confident answer even for images that are not skin photos.
- Performance depends on the size, balance, and quality of the training dataset, and may not generalise to all skin tones, lighting conditions, or camera types.
- The model has not been clinically validated.

## Technologies

Python, PyTorch, Hugging Face Transformers, scikit-learn, FastAPI, Uvicorn, Google Colab

## Author

Ali Jaffar, FAST-NUCES
