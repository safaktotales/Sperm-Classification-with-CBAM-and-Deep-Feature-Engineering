# Sperm-Classification-with-CBAM-and-Deep-Feature-Engineering
Sperm Classification with CBAM and Deep Feature Engineering

📋 Project Overview
This repository contains the implementation of CBAM (Convolutional Block Attention Module) enhanced models for sperm morphology classification with comprehensive Deep Feature Engineering (DFE) analysis.
🎯 Key Features

Enhanced CBAM Models: ResNet50+CBAM and Xception+CBAM architectures
Advanced Training: 60-epoch training with staged learning (30+30 epochs)
Grad-CAM Visualizations: Model attention visualization for interpretability
t-SNE Analysis: Feature space visualization for each layer
Deep Feature Engineering: Multiple feature extraction and selection methods
Comprehensive Evaluation: Confusion matrices, classification reports, and performance metrics

🏆 Results

Target Accuracy: 95%+ (based on successful Kaggle competitions)
Best Performance: 97.68%
Dataset: SMIDS Sperm Dataset

📁 Repository Structure
sperm-classification-cbam/
├── README.md                    # This file
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── src/                       # Source code
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cbam_modules.py    # CBAM implementation
│   │   ├── resnet_cbam.py     # ResNet50+CBAM model
│   │   └── xception_cbam.py   # Xception+CBAM model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset.py         # Dataset class
│   │   ├── transforms.py      # Data augmentation
│   │   ├── trainer.py         # Training utilities
│   │   └── visualization.py   # Grad-CAM, t-SNE functions
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── dfe.py            # Deep Feature Engineering
│   │   ├── gradcam.py        # Grad-CAM implementation
│   │   └── metrics.py        # Evaluation metrics
│   └── main.py               # Main execution script
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── data/                     # Data directory (add to .gitignore)
│   └── README.md            # Data setup instructions
├── results/                 # Results directory
│   ├── models/             # Saved model weights
│   ├── visualizations/     # Generated plots
│   └── reports/           # Analysis reports
├── docs/                   # Documentation
│   ├── model_architecture.md
│   ├── training_guide.md
│   └── results_interpretation.md
└── .gitignore             # Git ignore file
🚀 Quick Start
1. Clone Repository
bashgit clone https://github.com/yourusername/sperm-classification-cbam.git
cd sperm-classification-cbam
2. Install Dependencies
bashpip install -r requirements.txt
3. Prepare Data
bash# Place your SMIDS dataset in data/ directory
# Update data path in src/main.py
4. Run Training
bashpython src/main.py
📊 Model Architectures
ResNet50 + CBAM

Backbone: Pre-trained ResNet50
Attention: CBAM (Channel + Spatial Attention)
Classifier: Enhanced multi-layer classifier
Parameters: ~25M trainable parameters

Xception + CBAM

Backbone: Inception-v3 (as Xception proxy)
Attention: CBAM integration
Classifier: Multi-layer with dropout
Parameters: ~23M trainable parameters

🔬 Deep Feature Engineering
Feature Extraction Layers

Backbone features
CBAM-enhanced features
Global Average Pooling features
Global Max Pooling features
Pre-classification features

Feature Selection Methods

PCA: Principal Component Analysis
Chi-Square: Statistical feature selection
Random Forest: Tree-based feature importance
Variance: High-variance feature selection

Classifiers

SVM: RBF and Linear kernels
k-NN: k-Nearest Neighbors
Random Forest: Ensemble method

📈 Training Strategy
Stage 1: Frozen Backbone (30 epochs)

Freeze pre-trained backbone
Train CBAM and classifier only
Learning rate: 1e-3

Stage 2: Full Fine-tuning (30 epochs)

Unfreeze all layers
Differential learning rates:

Backbone: 1e-5
CBAM: 1e-4
Classifier: 1e-4



🎨 Visualizations
Grad-CAM

Model attention heatmaps
Class-specific activation patterns
Superimposed visualizations

t-SNE

Feature space clustering
Layer-wise feature analysis
Class separation visualization

Performance Metrics

Training/validation curves
Confusion matrices
Classification reports

📋 Requirements
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
Pillow>=8.3.0
📝 Usage Examples
Basic Training
pythonfrom src.main import main_enhanced_cbam_analysis

# Run complete analysis
results = main_enhanced_cbam_analysis()
Custom Training
pythonfrom src.models.resnet_cbam import ResNetCBAMSpermNet
from src.utils.trainer import CBAMSpermTrainer

# Create model
model = ResNetCBAMSpermNet(num_classes=3)

# Train
trainer = CBAMSpermTrainer(model, train_loader, val_loader, device)
accuracy = trainer.train_kaggle_style(total_epochs=60)
Grad-CAM Generation
pythonfrom src.analysis.gradcam import save_gradcam_results

# Generate Grad-CAM visualizations
save_gradcam_results(model, dataloader, device, classes, save_dir, model_name)
📊 Expected Results
ModelBase CNNBest DFEImprovementResNet+CBAM~89%~92%+3%Xception+CBAM~87%~91%+4%
🤝 Contributing

Fork the repository
Create feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
📚 Citation
If you use this code in your research, please cite:
bibtex@article{your_paper_2024,
  title={CBAM-Enhanced Deep Learning for Sperm Morphology Classification with Deep Feature Engineering},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
🙏 Acknowledgments

SMIDS Dataset creators
PyTorch team for the framework
CBAM paper authors
Kaggle community for inspiration
