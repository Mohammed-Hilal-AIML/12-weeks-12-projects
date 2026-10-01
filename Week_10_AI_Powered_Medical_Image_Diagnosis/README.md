# Week 10 — AI-Powered Medical Image Diagnosis

**Educational/research project only. Not a medical device and not for clinical diagnosis.**

PyTorch DenseNet121 transfer-learning project for binary chest X-ray classification (NORMAL vs PNEUMONIA), with class-imbalance handling, Grad-CAM explainability, clinical-style evaluation, and Monte Carlo Dropout uncertainty.

## Guide-aligned workflow

```text
Chest X-Ray dataset
      ↓
class imbalance analysis
      ↓
class weights + weighted sampling + augmentation
      ↓
DenseNet121 pretrained on ImageNet
      ↓
1024 → 256 → ReLU → Dropout → 1 → Sigmoid
      ↓
15 epochs, Adam 1e-4, ReduceLROnPlateau
      ↓
ROC-AUC + sensitivity + specificity + confusion matrix
      ↓
Grad-CAM
      ↓
20-pass Monte Carlo Dropout uncertainty
```

## Dataset

Use Kaggle **Chest X-Ray Images (Pneumonia)**. Do not upload the images to GitHub.

Place the extracted dataset at:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/                 # validation/ also supported
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Repository

```text
Week_10_AI_Powered_Medical_Image_Diagnosis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/README.md
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── uncertainty.py
├── models/.gitkeep
└── outputs/README.md
```

## Windows / Python 3.12 / no venv

```powershell
py -3.12 --version
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
py -3.12 -c "import torch, torchvision, cv2, sklearn; print('All dependencies OK')"
py -3.12 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Train

```powershell
py -3.12 scripts/train.py
```

The script uses DenseNet121, freezes the early feature layers, unfreezes the last dense block, uses class weighting and a weighted sampler, mixed precision when CUDA is available, Adam `1e-4`, ReduceLROnPlateau, and 15 epochs. The best validation-AUC model is saved as:

```text
models/densenet121_pneumonia_best.pth
```

## Evaluate

```powershell
py -3.12 scripts/evaluate.py
```

Reports ROC-AUC, sensitivity (pneumonia recall), specificity, accuracy, confusion matrix, and classification report.

The guide's sensitivity >90% is a **target**, not a guaranteed result. Never claim it unless your actual test run achieves it.

## Grad-CAM

```powershell
py -3.12 scripts/gradcam.py --image "data/chest_xray/test/PNEUMONIA/your_image.jpeg"
```

Output is saved under `outputs/gradcam/`.

## Monte Carlo Dropout

```powershell
py -3.12 scripts/uncertainty.py --image "data/chest_xray/test/PNEUMONIA/your_image.jpeg"
```

The script performs 20 stochastic passes, reports mean probability and standard deviation, and flags higher uncertainty using an educational threshold. This is not a clinical uncertainty threshold.

## Medical-AI limitations

Performance can change with dataset bias, label noise, image quality, acquisition differences, disease prevalence, and distribution shift. False negatives and false positives have different clinical consequences. Grad-CAM is an explanation aid, not proof that highlighted regions are medically meaningful. This project is for learning deep learning, explainability, evaluation, and responsible AI—not clinical use.

## GitHub

Repository name:

`Week_10_AI_Powered_Medical_Image_Diagnosis`

Description:

`PyTorch DenseNet121 chest X-ray pneumonia classifier with class-imbalance handling, Grad-CAM explainability, clinical-style evaluation, and Monte Carlo Dropout uncertainty.`
