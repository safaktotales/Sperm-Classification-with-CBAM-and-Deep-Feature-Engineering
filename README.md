# 🧬 CBAM-Enhanced ResNet50 for Sperm Morphology Classification

This repository contains the complete implementation of a high-accuracy sperm morphology classification pipeline using a CBAM-augmented ResNet50 model, along with deep feature engineering (DFE) and comprehensive visualizations. The work is designed to achieve expert-level accuracy on the SMIDS dataset and is aligned with the methodology described in our upcoming SCI-indexed article.

---

## 📌 Highlights

- ✅ Achieved **96.3% accuracy** on the SMIDS dataset.
- 🧠 Used **CBAM (Convolutional Block Attention Module)** to enhance spatial and channel-wise feature learning.
- 🔬 Integrated **Deep Feature Engineering (DFE)** with feature selection (PCA, Chi2, etc.) and classifiers (SVM, kNN, RF).
- 🎯 Visualized results with **GradCAM**, **t-SNE**, **PCA**, and **Confusion Matrix**.
- 📄 Fully reproducible training and evaluation pipeline using PyTorch.

---

## 📁 Dataset Used

- **SMIDS** (Sperm Morphology Image Dataset for Segmentation and Classification)
  - Classes: `Normal`, `Abnormal`, `Non-sperm`
  - Format: `.bmp` images
  - Organized in class-wise folders (e.g., `SMIDS/Normal/`, `SMIDS/Abnormal/`)

---

## 🚀 How to Run

```bash
# Clone this repository
git clone https://github.com/your-username/sperm-morphology-cbam.git
cd sperm-morphology-cbam

# (Optional) Create virtual environment
conda create -n sperm-cbam python=3.9
conda activate sperm-cbam

# Install dependencies
pip install -r requirements.txt

# Run main training & visualization script
python main_cbam_pipeline.py
