# Green Box Object Detection and Counting Assessment

## 1. Project Overview
This project implements an end-to-end pipeline for detecting and counting `green_box` objects using a **Faster R-CNN model with a ResNet-50 FPN backbone**. In strict adherence to assessment constraints, **no YOLO models or architecture variants were used**.

- **Task:** Object Detection & Precise Counting
- **Target Class:** `green_box`
- **Backbone Architecture:** Faster R-CNN with ResNet-50 Feature Pyramid Network (FPN)
- **Dataset Structure:** Pascal VOC XML annotations (70 train, 15 val, 15 test)

---

## 2. Quantitative Evaluation Metrics
Evaluated on the held-out test split (15 images) using `torchmetrics` (IoU threshold = 0.50):

| Metric | Baseline (10 Epochs) | Fine-Tuned (25 Epochs + Augmentation) |
|---|---|---|
| **mAP @ IoU 0.50** | 28.25% | **36.29%** |
| **Recall (mAR@100)** | 50.15% | **54.23%** |
| **F1-Score** | 36.14% | **43.48%** |

### Key Improvements Introduced:
1. **Data Augmentation:** Random Horizontal Flips and Color/Brightness Jitter to handle lighting variations.
2. **Anchor Box Optimization:** Custom anchor generator scales `(8, 16, 32, 64, 128)` for small and dense bounding boxes.
3. **Learning Rate Decay:** StepLR scheduler (0.005 -> 0.0005) at Epoch 15 for stable convergence.

---

## 3. Project Structure

.
├── dataset/              # Pascal VOC images & labels (train, val, test)
├── models/
│   └── model_weights.pth# Trained Faster R-CNN PyTorch weights
├── results/
│   ├── annotated_images/ #Visualized predictions with bounding boxes & counts
│   └── detection_results.json # Bounding box coordinates & object counts
├── dist/
│   └── green_box_detector-1.0.0-py3-none-any.whl # Built Python Wheel package
├── setup.py               # Packaging configuration script
├── train.py               # Training script with anchor optimization
├── evaluation.py          # Quantitative metric evaluation script
├── inference.py           # Batch inference & overlay visualizer script
├── evaluation_report.txt  # Saved text report of final evaluation metrics
├── requirements           # Dependency and packages
└── README.md              # Submission Documentation
|-- Output Sample          # annotated img for example
|-- prepare Dataset for
    tran, val and test
    using CVAT             # preparing DATASET
____________________________________________________________________________________________________________________________________________

## 4. To check on new images <------------------


1. Upload NEW img into test_images folder 
2. Run inference.py 
3. The results/ folder will be CREATED/UPDATED with annotated images and detection results in annotations.json format.