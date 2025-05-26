# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:02:40 2025

@author: MONSTER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from collections import Counter
import time
import pandas as pd

# Grad-CAM implementation
class GradCAM:
    """Grad-CAM visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_idx):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""

    def __init__(self, channels, ratio=8):
        super(CBAMBlock, self).__init__()
        self.channels = channels
        self.ratio = ratio

        # Channel Attention Module
        self.channel_attention = ChannelAttention(channels, ratio)
        # Spatial Attention Module
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Apply channel attention
        x = self.channel_attention(x)
        # Apply spatial attention
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    """Channel Attention Module"""

    def __init__(self, channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling branch
        avg_out = self.shared_mlp(self.avg_pool(x))
        # Max pooling branch
        max_out = self.shared_mlp(self.max_pool(x))
        # Combine and apply sigmoid
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate
        out = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution and sigmoid
        out = self.conv(out)
        return x * self.sigmoid(out)

class ResNetCBAMSpermNet(nn.Module):
    """ResNet50 + CBAM for Sperm Classification"""

    def __init__(self, num_classes=3, pretrained=True, cbam_ratio=8):
        super(ResNetCBAMSpermNet, self).__init__()

        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)

        # Remove the final classifier
        self.backbone.fc = nn.Identity()

        # Get feature dimension
        feature_dim = 2048  # ResNet50 feature dimension

        # Add CBAM attention
        self.cbam = CBAMBlock(feature_dim, ratio=cbam_ratio)

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim * 2),
            nn.Dropout(0.5),
            nn.Linear(feature_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Feature extraction points
        self.extraction_points = {
            'backbone_features': None,
            'cbam_features': None,
            'gap_features': None,
            'gmp_features': None,
            'pre_final_features': None
        }

        self.freeze_backbone()
        print(f"✅ ResNet50+CBAM Model initialized")
        print(f"📊 Feature dimension: {feature_dim}")
        print(f"🔧 Total parameters: {self.count_parameters():,}")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x, extract_features=False):
        # Extract features through backbone layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        backbone_features = self.backbone.layer4(x)

        if extract_features:
            self.extraction_points['backbone_features'] = backbone_features.detach()

        # Apply CBAM attention
        cbam_features = self.cbam(backbone_features)

        if extract_features:
            self.extraction_points['cbam_features'] = cbam_features.detach()

        # Global pooling
        gap_features = self.global_avg_pool(cbam_features).flatten(1)
        gmp_features = self.global_max_pool(cbam_features).flatten(1)

        if extract_features:
            self.extraction_points['gap_features'] = gap_features.detach()
            self.extraction_points['gmp_features'] = gmp_features.detach()

        # Combine features
        combined_features = torch.cat([gap_features, gmp_features], dim=1)

        if extract_features:
            self.extraction_points['pre_final_features'] = combined_features.detach()

        # Final classification
        output = self.classifier(combined_features)

        return output

    def get_features(self, dataloader, device, layer_names=['cbam_features', 'gap_features']):
        """Extract features from specified layers"""
        self.eval()
        features_dict = {name: [] for name in layer_names}
        labels = []

        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(device)
                _ = self.forward(data, extract_features=True)

                for layer_name in layer_names:
                    if layer_name in self.extraction_points and self.extraction_points[layer_name] is not None:
                        features = self.extraction_points[layer_name].cpu().numpy()
                        # Flatten if needed
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        features_dict[layer_name].append(features)

                labels.extend(target.numpy())

        # Convert to numpy arrays
        final_features = {}
        for name in layer_names:
            if features_dict[name]:
                final_features[name] = np.vstack(features_dict[name])

        return final_features, np.array(labels)

class XceptionCBAMSpermNet(nn.Module):
    """Xception-like + CBAM for Sperm Classification"""

    def __init__(self, num_classes=3, pretrained=True, cbam_ratio=8):
        super(XceptionCBAMSpermNet, self).__init__()

        # Use Inception V3 as Xception proxy
        self.backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
        self.backbone.fc = nn.Identity()

        feature_dim = 2048

        # Add CBAM attention
        self.cbam = CBAMBlock(feature_dim, ratio=cbam_ratio)

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim * 2),
            nn.Dropout(0.5),
            nn.Linear(feature_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Feature extraction points
        self.extraction_points = {
            'backbone_features': None,
            'cbam_features': None,
            'gap_features': None,
            'gmp_features': None,
            'pre_final_features': None
        }

        self.freeze_backbone()
        print(f"✅ Xception+CBAM Model initialized")
        print(f"📊 Feature dimension: {feature_dim}")
        print(f"🔧 Total parameters: {self.count_parameters():,}")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x, extract_features=False):
        # Backbone feature extraction
        backbone_features = self.backbone(x)

        # Reshape for CBAM if needed
        if backbone_features.dim() == 2:
            batch_size = backbone_features.size(0)
            feature_dim = backbone_features.size(1)
            backbone_features = backbone_features.view(batch_size, feature_dim, 1, 1)

        if extract_features:
            self.extraction_points['backbone_features'] = backbone_features.detach()

        # Apply CBAM attention
        cbam_features = self.cbam(backbone_features)

        if extract_features:
            self.extraction_points['cbam_features'] = cbam_features.detach()

        # Global pooling
        gap_features = self.global_avg_pool(cbam_features).flatten(1)
        gmp_features = self.global_max_pool(cbam_features).flatten(1)

        if extract_features:
            self.extraction_points['gap_features'] = gap_features.detach()
            self.extraction_points['gmp_features'] = gmp_features.detach()

        # Combine features
        combined_features = torch.cat([gap_features, gmp_features], dim=1)

        if extract_features:
            self.extraction_points['pre_final_features'] = combined_features.detach()

        # Final classification
        output = self.classifier(combined_features)

        return output

    def get_features(self, dataloader, device, layer_names=['cbam_features', 'gap_features']):
        """Extract features from specified layers"""
        self.eval()
        features_dict = {name: [] for name in layer_names}
        labels = []

        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(device)
                _ = self.forward(data, extract_features=True)

                for layer_name in layer_names:
                    if layer_name in self.extraction_points and self.extraction_points[layer_name] is not None:
                        features = self.extraction_points[layer_name].cpu().numpy()
                        if features.ndim > 2:
                            features = features.reshape(features.shape[0], -1)
                        features_dict[layer_name].append(features)

                labels.extend(target.numpy())

        final_features = {}
        for name in layer_names:
            if features_dict[name]:
                final_features[name] = np.vstack(features_dict[name])

        return final_features, np.array(labels)

# Dataset class
class SMIDSSpermDataset(Dataset):
    """SMIDS sperm dataset"""

    def __init__(self, root_dir, indices=None, transform=None, phase='train', target_balance=True):
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all data
        all_images = []
        all_labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_images = [f for f in os.listdir(class_dir) if f.lower().endswith('.bmp')]

            for img_name in class_images:
                all_images.append(os.path.join(class_dir, img_name))
                all_labels.append(self.class_to_idx[class_name])

        # Apply indices if provided
        if indices is not None:
            self.images = [all_images[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
        else:
            self.images = all_images
            self.labels = all_labels

        # Balance classes if requested
        if target_balance and phase == 'train':
            self.balance_dataset()

        self.print_distribution()

    def balance_dataset(self):
        """Balance dataset"""
        class_groups = {}
        for img, label in zip(self.images, self.labels):
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(img)

        class_counts = [len(images) for images in class_groups.values()]
        target_count = int(np.median(class_counts) * 1.2)

        balanced_images = []
        balanced_labels = []

        for class_idx, class_images in class_groups.items():
            current_count = len(class_images)

            balanced_images.extend(class_images)
            balanced_labels.extend([class_idx] * current_count)

            if current_count < target_count:
                needed = target_count - current_count
                additional_images = np.random.choice(class_images, needed, replace=True)
                balanced_images.extend(additional_images)
                balanced_labels.extend([class_idx] * needed)

        self.images = balanced_images
        self.labels = balanced_labels

    def print_distribution(self):
        class_counts = Counter(self.labels)
        total = len(self.labels)

        print(f"📊 {self.phase.upper()} Distribution ({total} total):")
        for i, class_name in enumerate(self.classes):
            count = class_counts[i]
            percentage = (count / total) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load and enhance image
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Sperm-specific enhancement
        image = self.enhance_sperm_features(image)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def enhance_sperm_features(self, image):
        """Enhanced sperm feature enhancement"""
        # CLAHE contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # Gentle denoising
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)

        # Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Blend
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

        return np.clip(result, 0, 255).astype(np.uint8)

def get_transforms(phase='train', img_size=224):
    """Get transforms for training/validation"""

    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Trainer class
class CBAMSpermTrainer:
    """Trainer for CBAM-based sperm classification"""

    def __init__(self, model, train_loader, val_loader, device, num_classes=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        self.best_acc = 0.0
        self.best_model_state = None

    def train_kaggle_style(self, total_epochs=60):  # 30+30=60 epochs
        """Train using Kaggle-style approach"""

        print(f"\n🚀  (TARGET: 91%+)")
        print("=" * 60)
        print(f"📋 Total epochs: {total_epochs}")

        # Stage 1: Train with frozen backbone (30 epochs)
        stage1_epochs = total_epochs // 2
        print(f"\n📋 STAGE 1: Frozen backbone ({stage1_epochs} epochs)")

        self.model.freeze_backbone()

        optimizer1 = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.001
        )

        for epoch in range(stage1_epochs):
            train_loss, train_acc = self.train_epoch(optimizer1)
            val_loss, val_acc = self.validate()

            self.update_history(train_loss, train_acc, val_loss, val_acc, optimizer1.param_groups[0]['lr'])

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f'Epoch {epoch+1}/{stage1_epochs}: 🎉 NEW BEST! Val: {val_acc:.2f}%')
            elif epoch % 5 == 0:
                print(f'Epoch {epoch+1}/{stage1_epochs}: Val: {val_acc:.2f}%')

        # Stage 2: Fine-tune with unfrozen backbone (30 epochs)
        stage2_epochs = total_epochs - stage1_epochs
        print(f"\n📋 STAGE 2: Full fine-tuning ({stage2_epochs} epochs)")

        self.model.unfreeze_backbone()

        optimizer2 = optim.Adam([
            {'params': self.model.backbone.parameters(), 'lr': 1e-5},
            {'params': self.model.cbam.parameters(), 'lr': 1e-4},
            {'params': self.model.classifier.parameters(), 'lr': 1e-4}
        ])

        for epoch in range(stage2_epochs):
            train_loss, train_acc = self.train_epoch(optimizer2)
            val_loss, val_acc = self.validate()

            self.update_history(train_loss, train_acc, val_loss, val_acc, optimizer2.param_groups[0]['lr'])

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f'Epoch {stage1_epochs + epoch+1}/{total_epochs}: 🎉 NEW BEST! Val: {val_acc:.2f}%')
            elif epoch % 5 == 0:
                print(f'Epoch {stage1_epochs + epoch+1}/{total_epochs}: Val: {val_acc:.2f}%')

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        print(f"\n🏆 TRAINING COMPLETED!")
        print(f"📊 Best Validation Accuracy: {self.best_acc:.2f}%")

        return self.best_acc

    def train_epoch(self, optimizer):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return total_loss / len(self.val_loader), 100. * correct / total

    def update_history(self, train_loss, train_acc, val_loss, val_acc, lr):
        """Update training history"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)

# Visualization functions
def save_gradcam_results(model, dataloader, device, classes, save_dir, model_name, num_samples=20):
    """Generate and save Grad-CAM visualizations"""
    
    print(f"🔍 Generating Grad-CAM for {model_name}...")
    
    # Create Grad-CAM instance
    if isinstance(model, ResNetCBAMSpermNet):
        target_layer = model.backbone.layer4[-1]  # Last conv layer of ResNet
    else:  # XceptionCBAMSpermNet
        target_layer = model.cbam  # CBAM layer
    
    gradcam = GradCAM(model, target_layer)
    
    # Get some samples from each class
    class_samples = {i: [] for i in range(len(classes))}
    
    for data, targets in dataloader:
        for i, (img, target) in enumerate(zip(data, targets)):
            class_idx = target.item()
            if len(class_samples[class_idx]) < num_samples // len(classes):
                class_samples[class_idx].append((img, class_idx))
        
        # Check if we have enough samples
        if all(len(samples) >= num_samples // len(classes) for samples in class_samples.values()):
            break
    
    # Generate Grad-CAM for selected samples
    gradcam_dir = os.path.join(save_dir, 'gradcam', model_name)
    os.makedirs(gradcam_dir, exist_ok=True)
    
    for class_idx, samples in class_samples.items():
        class_name = classes[class_idx]
        
        for sample_idx, (img, _) in enumerate(samples[:5]):  # Top 5 per class
            img_tensor = img.unsqueeze(0).to(device)
            
            # Generate CAM
            cam = gradcam.generate_cam(img_tensor, class_idx)
            
            # Original image
            orig_img = img.permute(1, 2, 0).numpy()
            orig_img = (orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            orig_img = np.clip(orig_img, 0, 1)
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(orig_img)
            plt.title(f'Original - {class_name}')
            plt.axis('off')
            
            # Grad-CAM heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(cam, cmap='jet')
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            
            # Superimposed
            plt.subplot(1, 3, 3)
            plt.imshow(orig_img)
            plt.imshow(cam, cmap='jet', alpha=0.4)
            plt.title('Superimposed')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(gradcam_dir, f'{class_name}_{sample_idx+1}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"✅ Grad-CAM saved to: {gradcam_dir}")

def save_tsne_visualization(features, labels, classes, save_dir, model_name, layer_name):
    """Generate and save t-SNE visualization"""
    
    print(f"📊 Generating t-SNE for {model_name} - {layer_name}...")
    
    # Reduce dimensionality first if needed
    if features.shape[1] > 50:
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(classes)]
    
    for i, class_name in enumerate(classes):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=colors[i], label=class_name, alpha=0.6, s=50)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f't-SNE Visualization - {model_name} ({layer_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    tsne_dir = os.path.join(save_dir, 'tsne')
    os.makedirs(tsne_dir, exist_ok=True)
    plt.savefig(os.path.join(tsne_dir, f'{model_name}_{layer_name}_tsne.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ t-SNE saved: {model_name}_{layer_name}_tsne.png")

def save_confusion_matrix(model, dataloader, device, classes, save_dir, model_name):
    """Generate and save confusion matrix"""
    
    print(f"📊 Generating confusion matrix for {model_name}...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save
    cm_dir = os.path.join(save_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    plt.savefig(os.path.join(cm_dir, f'{model_name}_confusion_matrix.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    # Save report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(cm_dir, f'{model_name}_classification_report.csv'))
    
    print(f"✅ Confusion matrix saved: {model_name}_confusion_matrix.png")
    
    return cm, report

def plot_training_history(history, save_dir, model_name):
    """Plot and save training history"""
    
    print(f"📈 Plotting training history for {model_name}...")
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.axhline(y=91, color='green', linestyle='--', alpha=0.7, label='Target (91%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3.plot(epochs, history['lr'], 'g-')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Performance metrics
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    
    metrics = ['Final Train Acc', 'Final Val Acc', 'Best Val Acc', 'Target (91%)']
    values = [final_train_acc, final_val_acc, best_val_acc, 91]
    colors = ['blue', 'red', 'green', 'orange']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_title('Performance Summary')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save
    history_dir = os.path.join(save_dir, 'training_history')
    os.makedirs(history_dir, exist_ok=True)
    plt.savefig(os.path.join(history_dir, f'{model_name}_training_history.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training history saved: {model_name}_training_history.png")

def run_dfe_analysis(model, train_loader, val_loader, device, classes, save_dir, model_name):
    """Run Deep Feature Engineering analysis"""
    
    print(f"🔬 Running DFE analysis for {model_name}...")
    
    # Feature extraction layers
    extraction_layers = ['cbam_features', 'gap_features', 'gmp_features', 'pre_final_features']
    
    # Feature selection methods
    feature_selectors = {
        'PCA': lambda X, y: PCA(n_components=min(256, X.shape[1], len(X)//2)).fit_transform(X),
        'Chi2': lambda X, y: SelectKBest(chi2, k=min(256, X.shape[1])).fit_transform(X - X.min() + 1e-8, y),
        'RF': lambda X, y: SelectKBest(f_classif, k=min(256, X.shape[1])).fit_transform(X, y),
        'Variance': lambda X, y: X[:, np.argsort(np.var(X, axis=0))[::-1][:min(256, X.shape[1])]],
    }
    
    # Classifiers
    classifiers = {
        'SVM_RBF': SVC(kernel='rbf', C=1.0, random_state=42),
        'SVM_Linear': SVC(kernel='linear', C=1.0, random_state=42),
        'kNN_3': KNeighborsClassifier(n_neighbors=3, metric='euclidean'),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    try:
        # Extract features
        features_dict, labels = model.get_features(train_loader, device, extraction_layers)
        
        if not features_dict:
            print("❌ No features extracted!")
            return None
        
        # Run DFE combinations
        dfe_results = []
        
        for layer_name, features in features_dict.items():
            print(f"\n📍 Processing layer: {layer_name} ({features.shape})")
            
            # Generate t-SNE visualization for this layer
            save_tsne_visualization(features, labels, classes, save_dir, model_name, layer_name)
            
            for selector_name, selector_func in feature_selectors.items():
                try:
                    # Apply feature selection
                    selected_features = selector_func(features, labels)
                    
                    for classifier_name, classifier in classifiers.items():
                        try:
                            # Train-test split for evaluation
                            X_train, X_test, y_train, y_test = train_test_split(
                                selected_features, labels, test_size=0.3, random_state=42,
                                stratify=labels
                            )
                            
                            # Train and evaluate
                            classifier.fit(X_train, y_train)
                            predictions = classifier.predict(X_test)
                            accuracy = np.mean(predictions == y_test) * 100
                            
                            dfe_results.append({
                                'layer': layer_name,
                                'selector': selector_name,
                                'classifier': classifier_name,
                                'accuracy': accuracy,
                                'method': f"{layer_name} + {selector_name} + {classifier_name}"
                            })
                            
                        except Exception as e:
                            print(f"  Error with {selector_name} + {classifier_name}: {str(e)}")
                            continue
                
                except Exception as e:
                    print(f"  Error with {selector_name}: {str(e)}")
                    continue
        
        # Save DFE results
        if dfe_results:
            dfe_results.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Save as CSV
            dfe_dir = os.path.join(save_dir, 'dfe_results')
            os.makedirs(dfe_dir, exist_ok=True)
            
            dfe_df = pd.DataFrame(dfe_results)
            dfe_df.to_csv(os.path.join(dfe_dir, f'{model_name}_dfe_results.csv'), index=False)
            
            # Plot top results
            top_10 = dfe_results[:10]
            
            plt.figure(figsize=(12, 8))
            methods = [r['method'] for r in top_10]
            accuracies = [r['accuracy'] for r in top_10]
            
            # Shorten method names for better visualization
            short_methods = [m.replace(' + ', '+\n') for m in methods]
            
            bars = plt.barh(range(len(short_methods)), accuracies, color='skyblue', alpha=0.8)
            plt.yticks(range(len(short_methods)), short_methods, fontsize=8)
            plt.xlabel('Accuracy (%)')
            plt.title(f'Top 10 DFE Results - {model_name}')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{acc:.1f}%', ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(dfe_dir, f'{model_name}_top_dfe_results.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ DFE analysis completed. Best: {dfe_results[0]['accuracy']:.2f}%")
            
            return dfe_results
    
    except Exception as e:
        print(f"❌ DFE Error: {str(e)}")
        return None

def main_enhanced_cbam_analysis():
    """Main function with enhanced analysis and visualizations"""
    
    print("🚀 ENHANCED CBAM SPERM CLASSIFICATION WITH VISUALIZATIONS")
    print("🎯 Target: 91%+ accuracy with comprehensive analysis")
    print("=" * 70)
    
    # Paths - Updated for your system
    data_path = "D:/SMIDS"  # Your data path
    save_dir = "D:/makale_sonuclari"  # Results directory
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📁 Dataset: {data_path}")
    print(f"💾 Results will be saved to: {save_dir}")
    print(f"💻 Device: {device}")
    
    # Prepare data
    all_images = []
    all_labels = []
    classes = sorted(os.listdir(data_path))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for class_name in classes:
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        class_images = [f for f in os.listdir(class_dir) if f.lower().endswith('.bmp')]
        
        for img_name in class_images:
            all_images.append(os.path.join(class_dir, img_name))
            all_labels.append(class_to_idx[class_name])
    
    print(f"📊 Total images: {len(all_images)}")
    print(f"🏷️  Classes: {classes}")
    
    # Split data
    train_idx, val_idx = train_test_split(
        range(len(all_images)),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )
    
    # Create datasets
    train_dataset = SMIDSSpermDataset(
        data_path,
        indices=train_idx,
        transform=get_transforms('train', 224),
        phase='train',
        target_balance=True
    )
    
    val_dataset = SMIDSSpermDataset(
        data_path,
        indices=val_idx,
        transform=get_transforms('val', 224),
        phase='val',
        target_balance=False
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    # Test multiple models
    models_to_test = [
        ('ResNetCBAM', ResNetCBAMSpermNet),
        ('XceptionCBAM', XceptionCBAMSpermNet)
    ]
    
    all_results = {}
    
    for model_name, model_class in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧠 TESTING: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = model_class(num_classes=len(classes), pretrained=True)
            
            # Create trainer
            trainer = CBAMSpermTrainer(model, train_loader, val_loader, device, len(classes))
            
            # Train model
            print(f"🚂 Training {model_name} for 60 epochs (30+30)...")
            start_time = time.time()
            base_accuracy = trainer.train_kaggle_style(total_epochs=60)  # 30+30=60
            training_time = time.time() - start_time
            
            # Plot training history
            plot_training_history(trainer.history, save_dir, model_name)
            
            # Generate confusion matrix
            cm, report = save_confusion_matrix(model, val_loader, device, classes, save_dir, model_name)
            
            # Generate Grad-CAM visualizations
            save_gradcam_results(model, val_loader, device, classes, save_dir, model_name)
            
            # Run DFE analysis with t-SNE
            dfe_results = run_dfe_analysis(model, train_loader, val_loader, device, classes, save_dir, model_name)
            
            all_results[model_name] = {
                'base_accuracy': base_accuracy,
                'training_time': training_time,
                'model_params': model.count_parameters(),
                'dfe_results': dfe_results,
                'confusion_matrix': cm,
                'classification_report': report
            }
            
            print(f"✅ {model_name} completed!")
            print(f"   Base CNN: {base_accuracy:.2f}%")
            if dfe_results:
                print(f"   Best DFE: {dfe_results[0]['accuracy']:.2f}%")
                print(f"   Best Method: {dfe_results[0]['method']}")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
            continue
    
    # Generate final comparison report
    print(f"\n{'='*70}")
    print("📊 GENERATING FINAL COMPARISON REPORT")
    print(f"{'='*70}")
    
    # Create comprehensive comparison
    comparison_data = []
    
    for model_name, results in all_results.items():
        base_acc = results['base_accuracy']
        best_dfe_acc = results['dfe_results'][0]['accuracy'] if results['dfe_results'] else base_acc
        best_method = results['dfe_results'][0]['method'] if results['dfe_results'] else 'Base CNN'
        
        comparison_data.append({
            'Model': model_name,
            'Base_CNN_Accuracy': base_acc,
            'Best_DFE_Accuracy': best_dfe_acc,
            'Best_DFE_Method': best_method,
            'Improvement': best_dfe_acc - base_acc,
            'Parameters_M': results['model_params'] / 1e6,
            'Training_Time_min': results['training_time'] / 60
        })
    
    # Save comparison as CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(save_dir, 'final_comparison.csv'), index=False)
    
    # Create final comparison plot
    if len(comparison_data) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        models = [d['Model'] for d in comparison_data]
        base_accs = [d['Base_CNN_Accuracy'] for d in comparison_data]
        dfe_accs = [d['Best_DFE_Accuracy'] for d in comparison_data]
        
        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, base_accs, width, label='Base CNN', alpha=0.8, color='lightblue')
        ax1.bar(x + width/2, dfe_accs, width, label='Best DFE', alpha=0.8, color='lightgreen')
        ax1.axhline(y=91, color='red', linestyle='--', alpha=0.7, label='Target (91%)')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement analysis
        improvements = [d['Improvement'] for d in comparison_data]
        bars = ax2.bar(models, improvements, color='orange', alpha=0.8)
        ax2.set_title('DFE Improvement over Base CNN')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Improvement (%)')
        ax2.grid(True, alpha=0.3)
        
        for bar, imp in zip(bars, improvements):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{imp:+.1f}%', ha='center', va='bottom')
        
        # Parameters vs Performance
        params = [d['Parameters_M'] for d in comparison_data]
        ax3.scatter(params, dfe_accs, s=100, alpha=0.7, c=['blue', 'red'][:len(models)])
        ax3.set_xlabel('Parameters (Millions)')
        ax3.set_ylabel('Best Accuracy (%)')
        ax3.set_title('Parameters vs Performance')
        ax3.grid(True, alpha=0.3)
        
        for i, model in enumerate(models):
            ax3.annotate(model, (params[i], dfe_accs[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Training time
        times = [d['Training_Time_min'] for d in comparison_data]
        bars = ax4.bar(models, times, color='purple', alpha=0.7)
        ax4.set_title('Training Time')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Time (minutes)')
        ax4.grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars, times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{time_val:.1f}min', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'final_comparison_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print final summary
    print(f"\n🏆 FINAL RESULTS SUMMARY")
    print("=" * 50)
    
    best_overall = None
    best_accuracy = 0
    
    for model_name, results in all_results.items():
        final_acc = results['dfe_results'][0]['accuracy'] if results['dfe_results'] else results['base_accuracy']
        
        print(f"\n📊 {model_name}:")
        print(f"   Base CNN: {results['base_accuracy']:.2f}%")
        if results['dfe_results']:
            print(f"   Best DFE: {results['dfe_results'][0]['accuracy']:.2f}%")
            print(f"   Best Method: {results['dfe_results'][0]['method']}")
            print(f"   Improvement: {results['dfe_results'][0]['accuracy'] - results['base_accuracy']:+.2f}%")
        print(f"   Parameters: {results['model_params']/1e6:.1f}M")
        print(f"   Training Time: {results['training_time']/60:.1f} minutes")
        
        if final_acc > best_accuracy:
            best_accuracy = final_acc
            best_overall = model_name
    
    print(f"\n🎯 BEST OVERALL RESULT:")
    print(f"   Model: {best_overall}")
    print(f"   Accuracy: {best_accuracy:.2f}%")
    print(f"   Target: 91.00%")
    print(f"   Difference: {best_accuracy - 91.0:+.2f}%")
    
    if best_accuracy >= 91:
        print("🎉 SUCCESS! Target achieved!")
    elif best_accuracy >= 88:
        print("🔥 EXCELLENT! Very close to target!")
    elif best_accuracy >= 85:
        print("📈 GREAT! Strong performance!")
    
    print(f"\n💾 All results saved to: {save_dir}")
    print("📁 Generated files:")
    print("   - Training histories")
    print("   - Grad-CAM visualizations")
    print("   - t-SNE plots")
    print("   - Confusion matrices")
    print("   - DFE analysis results")
    print("   - Final comparison report")
    
    return all_results

if __name__ == "__main__":
    print("🧬 ENHANCED CBAM SPERM CLASSIFICATION")
    print("🎯 60 epochs (30+30) training with comprehensive analysis")
    print("=" * 60)
    
    # Run the enhanced analysis
    results = main_enhanced_cbam_analysis()
    
    print(f"\n✅ Enhanced CBAM analysis completed!")
    print(f"🎊 Check 'D:/makale_sonuclari' for all generated files!")