# 🫁 AI-Powered Early Pediatric Pneumonia Detection

### Deep Learning for Chest X-Ray Analysis | University of Saida, Algeria

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.x-red?style=for-the-badge&logo=keras)](https://keras.io)
[![Google Colab](https://img.shields.io/badge/Google_Colab-GPU-yellow?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com)
[![Sensitivity](https://img.shields.io/badge/Sensitivity-97.21%25-success?style=for-the-badge)](https://github.com/labaninabila193-code/AI-Pediatric-Pneumonia-Detection)
[![Accuracy](https://img.shields.io/badge/Accuracy-94.31%25-success?style=for-the-badge)](https://github.com/labaninabila193-code/AI-Pediatric-Pneumonia-Detection)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

### AI-Powered Early Pediatric Pneumonia Detection: Integration with Electronic Medical Records in Algeria

*Leveraging Transfer Learning, Threshold Optimization, GradCAM Interpretability, and External Validation*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [My Role in This Project](#-my-role-in-this-project)
- [Full Team](#-full-team)
- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Model Comparison](#-model-comparison)
- [External Validation](#-external-validation)
- [Extension: COVID-19 vs Pneumonia Differentiation](#-extension-covid-19-vs-pneumonia-differentiation)
- [Grad-CAM Visualizations](#-grad-cam-visualizations)
- [Visual Examples](#-visual-examples)
- [Project Structure](#-project-structure)
- [Pre-trained Models](#-pre-trained-models)
- [How to Run](#-how-to-run)
- [Full Project Repository](#-full-project-repository)
- [References](#-references)
- [License](#-license)

---

## 🔍 Overview

This repository contains the **Deep Learning & Model Training** component of a complete, end-to-end AI-powered clinical decision support system for pediatric pneumonia detection.

Three pre-trained CNN architectures (VGG16, ResNet50, DenseNet121) were trained using transfer learning, evaluated with ROC-based threshold optimization, and validated on a fully independent external dataset. **Grad-CAM visualizations** were implemented to ensure clinical interpretability and radiologist trust.

> **Key Achievement:** DenseNet121 achieved **95.01% Sensitivity** and **94.31% Accuracy** on the internal test set, improving to **97.21% Sensitivity** on a completely independent external dataset — demonstrating strong real-world generalization.

---

## 🤖 My Role in This Project

**Labani Nabila Nour El Houda — Deep Learning Engineer**

This repository covers my specific contribution to the team project:

- ✅ Transfer learning pipeline across **VGG16, ResNet50, and DenseNet121**
- ✅ **Threshold optimization** via ROC analysis (default 0.5 → optimized 0.260), reducing missed pneumonia cases by **50%**
- ✅ **GradCAM clinical interpretability** — heatmaps showing which lung regions influenced each AI decision
- ✅ **External cross-dataset validation** on 488 independent images — achieving **97.21% sensitivity**
- ✅ **COVID-19 vs Pneumonia extension** — 3-class DenseNet121 classifier (92.41% overall accuracy)
- ✅ Integration of the trained DenseNet121 model into the team's full clinical Streamlit application

> The data preprocessing pipeline (70/15/15 stratified split, class weights, augmentation, TF data loaders) was developed by [@AminaMar](https://github.com/AminaMar/pediatric-pneumonia-detection) (Bouhmidi Amina Meroua). This project builds directly on that work for model training and evaluation.

---

## 👥 Full Team

| Role | Name | Core Responsibilities |
|------|------|-----------------------|
| 🔧 **Data Engineer** | **Bouhmidi Amina Meroua** | Complete data pipeline (EDA → preprocessing → data loaders), 70/15/15 stratified split, class weight computation, augmentation design |
| 🤖 **Deep Learning Engineer** | **Labani Nabila Nour El Houda** *(this repo)* | DenseNet121 / VGG16 / ResNet50 training, threshold optimization (0.260), GradCAM, external validation (97.21% sensitivity), COVID-19 extension |
| 💻 **Developer & Business Model** | **Miloudi Maroua Amira** | Full-stack Streamlit clinical app, RAG-based clinical reasoning (FAISS + BM25), automated PDF report generation, business model & deployment strategy |
| 📋 **Manager** | **Kassouar Fatima** | Gradient Boosting classifier (vital signs), CSV preprocessing pipeline and management |
| 👨‍🏫 Academic Supervisor | Dr. Abderrahmane Khiat | Academic guidance & evaluation |
| 🏥 Medical Advisor | Dr. Aimer Mohammed Djamel Eddine | Clinical validation, medical references & advisory |

**Institution:** University of Saida, Algeria — Academic Year 2025–2026

> 📂 **Full project repository (all pillars combined):** [github.com/AminaMar/pediatric-pneumonia-detection](https://github.com/AminaMar/pediatric-pneumonia-detection)

---

## 🚨 Problem Statement

Pneumonia is a **leading cause of death** in children under 5:

- 📊 **740,000+ deaths** annually worldwide (15% of child mortality under five)
- 🏥 **200–300 cases daily** in Algerian emergency departments, with 6–24 hour diagnostic delays
- 👨‍⚕️ **~1 pediatric radiologist per 500,000 children** — a 500× deficiency vs. developed nations
- 🚗 **40% of children** must travel 50–100 km to access chest X-ray imaging
- 📉 **70–85% radiologist agreement** on pneumonia diagnosis — significant inter-observer variability

---

## 💡 Proposed Solution

A deep learning system that:

- Automatically analyzes pediatric chest X-rays to detect pneumonia
- Achieves **≥95% sensitivity** — critical for medical screening where missing sick patients is dangerous
- Provides **Grad-CAM heatmaps** showing which lung regions influenced each AI decision
- Designed for integration with Algerian hospital Electronic Medical Records (DEM) via **HL7/FHIR** standards
- Deployed as part of a complete **Streamlit clinical application** built by the team

---

## 📁 Dataset

### Primary Training Dataset

- **Source:** [Kaggle Pediatric Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** 5,863 labeled images (1,583 NORMAL, 4,273 PNEUMONIA)
- **Patients:** Pediatric patients aged 1–5, Guangzhou Women and Children's Medical Center
- **Split:** 70% train / 15% validation / 15% test (stratified, 27%/73% class balance maintained)
- **Preprocessing:** Class weights {NORMAL: 1.850, PNEUMONIA: 0.685}, augmentation (rotation ±15°, shift 10%, zoom 10%, horizontal flip only), normalized to [0–1] at 224×224

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Training | 1,108 (27.0%) | 2,991 (73.0%) | 4,099 |
| Validation | 237 (27.0%) | 641 (73.0%) | 878 |
| Test | 238 (27.1%) | 641 (72.9%) | 879 |

### External Validation Dataset

- **Source:** [Pneumonia Radiography Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset)
- **Size:** 488 images (237 NORMAL, 251 PNEUMONIA)
- **Purpose:** Cross-dataset validation to test real-world generalization on completely unseen data

---

## 🔬 Methodology

```
Preprocessed Data (from Data Engineer — @AminaMar)
       ↓
Transfer Learning (ImageNet pretrained weights)
       ↓
Fine-tuning with Computed Class Weights {0: 1.850, 1: 0.685}
       ↓
Threshold Optimization via ROC Analysis (0.5 → 0.260)
       ↓
Evaluation (Sensitivity, Specificity, AUC-ROC, F1)
       ↓
GradCAM Clinical Interpretability
       ↓
External Dataset Validation (488 independent images)
       ↓
COVID-19 Extension (3-class DenseNet121)
```

### Models Trained

| # | Architecture | Parameters | ImageNet Weights | Training Time |
|---|-------------|------------|-----------------|--------------|
| 1 | VGG16 | 138M | ✅ | 120 min |
| 2 | ResNet50 | 25M | ✅ | ~60 min |
| 3 | **DenseNet121** ⭐ | **8M** | ✅ | **58 min** |

### Training Configuration

- **Platform:** Google Colab (T4 GPU)
- **Framework:** TensorFlow 2.19 / Keras 3.x
- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Early Stopping:** Patience = 7 epochs
- **Batch Size:** 32
- **Class Weights:** {0: 1.850, 1: 0.685} — provided by Data Engineer

---

## 📊 Results

### 🥇 DenseNet121 — Best Model ⭐

| Metric | Default Threshold (0.5) | **Optimized Threshold (0.260)** | Clinical Target |
|--------|------------------------|--------------------------------|----------------|
| **Accuracy** | 92.04% | **94.31%** | ≥90% ✅ |
| **Sensitivity** | 90.02% | **95.01%** | ≥95% ✅ |
| **Specificity** | 97.48% | **92.44%** | ≥85% ✅ |
| **Precision** | 98.97% | 97.13% | — |
| **F1-Score** | 0.9428 | — | — |
| **AUC-ROC** | **0.9810** | — | — |
| **Training Time** | — | **57.97 min** | — |

**Confusion Matrix (Optimized Threshold = 0.260):**

```
                    Predicted NORMAL    Predicted PNEUMONIA
Actual NORMAL            220 (TN)              18 (FP)
Actual PNEUMONIA          32 (FN)             609 (TP)
```

> ✅ **Clinical Impact:** Missed pneumonia cases reduced by **50%** (64 → 32) after threshold optimization.

---

### 🥈 VGG16

| Metric | Default (0.5) | Optimized (0.110) | Target |
|--------|--------------|-------------------|--------|
| Accuracy | 76.11% | 91.35% | ≥90% ✅ |
| Sensitivity | 67.39% | 95.32% | ≥95% ✅ |
| Specificity | 99.58% | 80.67% | ≥85% ⚠️ |
| AUC-ROC | 0.9644 | — | — |

> ⚠️ Required an extreme threshold (0.110) to reach sensitivity target. Specificity remains below clinical threshold.

---

### 🥉 ResNet50

| Metric | Default (0.5) | Optimized | Target |
|--------|--------------|-----------|--------|
| Accuracy | 82.14% | 78.16% | ≥90% ❌ |
| Sensitivity | 84.87% | 95.48% | ≥95% ✅ |
| Specificity | — | 31.51% | ≥85% ❌ |
| AUC-ROC | 0.8802 | — | — |

> ❌ **Clinically unacceptable:** 31.51% specificity means 69% of healthy children are incorrectly flagged as pneumonia cases.

---

## 🏆 Model Comparison

| Model | Accuracy | Sensitivity | Specificity | AUC-ROC | Time | Clinical Verdict |
|-------|----------|-------------|-------------|---------|------|-----------------|
| **DenseNet121** ⭐ | **94.31%** | **95.01%** | **92.44%** | **0.981** | **58 min** | ✅ **ALL TARGETS MET** |
| VGG16 | 91.35% | 95.32% | 80.67% | 0.964 | 120 min | ⚠️ Partial |
| ResNet50 | 78.16% | 95.48% | 31.51% | 0.880 | ~60 min | ❌ Not suitable |

### Why DenseNet121 Won

- ✅ **Only model meeting ALL clinical targets simultaneously**
- ✅ **Dense connections** — superior feature reuse, better gradient flow through deep layers
- ✅ **Parameter efficiency** — 8M vs 138M (VGG16) → significantly less overfitting risk
- ✅ **Clinically validated architecture** — Stanford CheXNet (2017) used DenseNet121 for chest X-rays
- ✅ **Fastest training** — 58 min vs 120 min for VGG16
- ✅ **Best AUC-ROC** — 0.981
- ✅ **Balanced performance** — no extreme threshold manipulation required (0.260 vs VGG16's 0.110)

> 💡 **Clinical Bottom Line:** DenseNet121 is the only model deployable in a real hospital setting without causing either missed pneumonia cases (low sensitivity) or overwhelming false alarms (low specificity).

---

## 🌍 External Validation

DenseNet121 was tested on a **completely independent dataset** from a different source (488 images, never seen during training or validation):

| Metric | Internal Test Set | External Dataset | Difference |
|--------|------------------|-----------------|------------|
| **Accuracy** | 94.31% | 87.09% | -7.22% |
| **Sensitivity** | 95.01% | **97.21%** | **+2.20%** ✅ |
| **Specificity** | 92.44% | 76.37% | -16.07% |
| Total Samples | 879 | 488 | — |

> 🎯 **Key Finding:** Sensitivity **improved** on the external dataset (97.21% vs 95.01%), confirming the model generalizes to real-world, unseen clinical data. Accuracy above 80% on a fully independent dataset is considered strong generalization in medical AI literature.

---

## 🦠 Extension: COVID-19 vs Pneumonia Differentiation

Following feedback from our academic supervisor, the binary model was extended to answer: **"Can the model differentiate a COVID-19 patient from a Pneumonia patient?"**

### Approach

- Fine-tuned DenseNet121 as a **3-class classifier**: COVID / Viral Pneumonia / Normal
- Dataset: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- 1,200 balanced images per class

### Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **COVID** | 90.6% | 91.1% | 90.9% |
| **Normal** | 88.5% | 90.0% | 89.3% |
| **Viral Pneumonia** | 98.3% | 96.1% | 97.2% |
| **Overall Accuracy** | | **92.41%** | |

### Key Finding

- COVID misclassified as Pneumonia: **only 2 cases out of 180**
- Pneumonia misclassified as COVID: **0 cases**
- The model successfully distinguishes between the two conditions

![Confusion Matrix](results/covid_extensions/confusion_matrix_multiclass(1).png)

---

## 🔥 Grad-CAM Visualizations

Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented on DenseNet121 to provide **clinical interpretability** — showing exactly which lung regions the AI focused on when making predictions.

The heatmaps confirm the model correctly focuses on:
- **Pneumonia cases:** Consolidation, infiltrates, and infected lung regions
- **Normal cases:** Central mediastinal structures (heart, trachea, diaphragm)

> This interpretability layer is critical for clinical trust — doctors can verify the AI is looking at the right anatomical regions before acting on predictions.

---

## 📸 Visual Examples

### ⚠️ PNEUMONIA Case — Correctly Detected

![Pneumonia Detection](results/gradcam_visualizations/gradcam_1_PNEUMONIA.png)

**Model Decision:** PNEUMONIA (Confidence: 99.3%)
**Grad-CAM:** Highlights consolidation in right lower lobe — typical bacterial pneumonia presentation
**Clinical Correlation:** ✅ Correct

---

### ✅ NORMAL Case — Correctly Classified

![Normal Classification](results/gradcam_visualizations/gradcam_4_NORMAL.png)

**Model Decision:** NORMAL (Confidence: 98.8%)
**Grad-CAM:** Focuses on central mediastinal structures, no pathological findings
**Clinical Correlation:** ✅ Correct

---

### 🔍 Additional Examples

| Pneumonia Cases | Normal Cases |
|----------------|--------------|
| ![](results/gradcam_visualizations/gradcam_3_PNEUMONIA.png) | ![](results/gradcam_visualizations/gradcam_5_NORMAL.png) |
| ![](results/gradcam_visualizations/gradcam_2_PNEUMONIA.png) | ![](results/gradcam_visualizations/gradcam_6_NORMAL.png) |

> 🩺 **Clinical Interpretation:** Heatmaps confirm DenseNet121 has learned clinically meaningful features — consolidation patterns for pneumonia, expected anatomical landmarks for normal cases.

---

## 📁 Project Structure

```
AI-Pediatric-Pneumonia-Detection/
│
├── 📓 Notebooks/
│   ├── 01_DenseNet121_Training.ipynb
│   ├── 01_DenseNet121_GradCAM_and_Threshold.ipynb
│   ├── 02_ResNet50_Training.ipynb
│   ├── 03_VGG16_Training.ipynb
│   ├── 04_External_Validation.ipynb
│   ├── 05_GradCAM.ipynb
│   └── 06_COVID_vs_Pneumonia_Classification.ipynb
│
├── 📊 results/
│   ├── densenet121/
│   │   ├── evaluation_report.txt
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── training_history.png
│   │   └── threshold_analysis.png
│   ├── resnet50/
│   ├── vgg16/
│   ├── gradcam_visualizations/
│   │   └── gradcam_1_PNEUMONIA.png ... gradcam_6_NORMAL.png
│   ├── covid_extensions/
│   │   └── confusion_matrix_multiclass.png
│   └── external_validation/
│       └── report.txt
│
├── 💾 saved_models/
│   ├── densenet121_best_model.keras   (34.3 MB) ⭐
│   ├── resnet50_best_model.keras      (102.6 MB)
│   └── vgg16_best_model.keras         (57.7 MB)
│
├── 📁 logs/
│   ├── densenet121_20260222_125840/
│   ├── resnet50_20260224_193657/
│   └── vgg16_20260225_113134/
│
├── 📁 docs/
├── 📋 README.md
└── 📄 LICENSE
```

---

## 💾 Pre-trained Models

Model files exceed GitHub's 25MB limit and are hosted on Google Drive.

👉 [**Download All Models (Google Drive)**](https://drive.google.com/drive/folders/1JtnqNL4lMSRHBtR_eex96k64wSix97Y9?usp=sharing)

| Model | Accuracy | Sensitivity | AUC-ROC | Size | Use |
|-------|----------|-------------|---------|------|-----|
| **densenet121_best_model.keras** | **94.31%** | **95.01%** | **0.981** | 34.3 MB | ⭐ Recommended |
| vgg16_best_model.keras | 91.35% | 95.32% | 0.964 | 57.7 MB | ⚠️ Acceptable |
| resnet50_best_model.keras | 78.16% | 95.48% | 0.880 | 102.6 MB | ❌ Not recommended |

```python
import tensorflow as tf

# Load the best model
model = tf.keras.models.load_model('saved_models/densenet121_best_model.keras')

# Predict on a new X-ray (preprocessed to 224x224, normalized [0-1])
prediction = model.predict(preprocessed_image)
# Output > 0.260 → PNEUMONIA;  Output ≤ 0.260 → NORMAL
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn opencv-python
```

### 1. Clone the Repository

```bash
git clone https://github.com/labaninabila193-code/AI-Pediatric-Pneumonia-Detection.git
cd AI-Pediatric-Pneumonia-Detection
```

### 2. Download the Dataset

- [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Place in: `data/chest_xray/train`, `data/chest_xray/val`, `data/chest_xray/test`

### 3. Open in Google Colab

All notebooks are designed for **Google Colab with T4 GPU**:
- Mount your Google Drive
- Update the data paths in Cell 2 of each notebook
- Run all cells in order

### 4. Training Order

```
01_DenseNet121_Training.ipynb        → Train & evaluate DenseNet121
02_ResNet50_Training.ipynb           → Train & evaluate ResNet50
03_VGG16_Training.ipynb              → Train & evaluate VGG16
04_External_Validation.ipynb         → Cross-dataset validation
05_GradCAM.ipynb                     → Interpretability visualizations
06_COVID_vs_Pneumonia_Classification → COVID-19 extension (3-class)
```

---

## 🔗 Full Project Repository

This repository covers **Pillar 2 (Deep Learning)** of a 4-pillar project. The complete system includes:

| Pillar | Lead | Description |
|--------|------|-------------|
| 🔧 Data Engineering | Bouhmidi Amina Meroua | EDA, preprocessing, data loaders |
| 🤖 **Deep Learning** *(this repo)* | **Labani Nabila Nour El Houda** | Model training, GradCAM, external validation |
| 💻 Clinical Application + RAG | Miloudi Maroua Amira | Streamlit app, RAG chatbot, PDF reports |
| 🌿 Vital Signs ML | Kassouar Fatima | Gradient Boosting classifier |

📂 **Full integrated project:** [github.com/AminaMar/pediatric-pneumonia-detection](https://github.com/AminaMar/pediatric-pneumonia-detection)

---

## 📚 References

1. **CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning** — *Stanford AI Lab (Rajpurkar et al., 2017)*
2. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization** — *Selvaraju et al., 2017*
3. **Detection of Pneumonia in Children Through Chest Radiographs Using AI in Low-Resource Settings** — *PLOS Digital Health, 2025*
4. **AI–EHR Integration Improving Diagnostic Capabilities Through HL7/FHIR Standards** — *PMC/PubMed Central, 2024*
5. **Diagnostic Performance of a Deep Learning Model Deployed at a National COVID-19 Screening Facility** — *Healthcare MDPI, 2022*
6. **Pneumonia in Children — Fact Sheet** — *WHO, 2024*
7. **IDSA/ATS Consensus Guidelines on Community-Acquired Pneumonia** — *Infectious Diseases Society of America*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Deep Learning Engineer:** Labani Nabila Nour El Houda — [GitHub @labaninabila193-code](https://github.com/labaninabila193-code)

**Full Team Repository:** [github.com/AminaMar/pediatric-pneumonia-detection](https://github.com/AminaMar/pediatric-pneumonia-detection)

---

**University of Saida, Algeria — 2025–2026**

*This project addresses a critical healthcare challenge in Algeria through state-of-the-art AI, aligned with the National Digital Health Strategy.*

*Complete pipeline: Data Engineering · Deep Learning · Clinical Application · Project Management*
