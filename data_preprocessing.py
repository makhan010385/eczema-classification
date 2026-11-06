import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class EczemaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label

def get_image_paths_and_labels(limit_per_class=50):
    """Get limited number of image paths and their corresponding labels"""
    image_paths = []
    labels = []
    
    # Get Eczema images (limited)
    eczema_files = [f for f in os.listdir(config.ECZEMA_PATH) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    eczema_files = eczema_files[:limit_per_class]  # Limit to specified number
    
    for file in eczema_files:
        image_paths.append(os.path.join(config.ECZEMA_PATH, file))
        labels.append(1)  # Eczema = 1
    
    # Get Normal images (limited)
    normal_files = [f for f in os.listdir(config.NORMAL_PATH) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    normal_files = normal_files[:limit_per_class]  # Limit to specified number
    
    for file in normal_files:
        image_paths.append(os.path.join(config.NORMAL_PATH, file))
        labels.append(0)  # Normal = 0
    
    print(f"📊 Using {len(eczema_files)} eczema images and {len(normal_files)} normal images")
    return image_paths, labels

def get_transforms(mode='train'):
    """Get data augmentation transforms"""
    if mode == 'train':
        transform = A.Compose([
            A.Resize(height=config.IMG_SIZE[0], width=config.IMG_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(height=config.IMG_SIZE[0], width=config.IMG_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform

def create_data_loaders(limit_per_class=50):
    """Create train, validation, and test data loaders with limited dataset"""
    # Get limited data
    image_paths, labels = get_image_paths_and_labels(limit_per_class)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, 
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=42,
        stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.TEST_RATIO/(config.VAL_RATIO + config.TEST_RATIO),
        random_state=42,
        stratify=y_temp
    )
    
    # Create datasets
    train_dataset = EczemaDataset(X_train, y_train, get_transforms('train'))
    val_dataset = EczemaDataset(X_val, y_val, get_transforms('val'))
    test_dataset = EczemaDataset(X_test, y_test, get_transforms('test'))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, (X_train, X_val, X_test, y_train, y_val, y_test)

def analyze_dataset():
    """Analyze and visualize dataset characteristics"""
    image_paths, labels = get_image_paths_and_labels()
    
    print(f"Total images: {len(image_paths)}")
    print(f"Eczema images: {sum(labels)}")
    print(f"Normal images: {len(labels) - sum(labels)}")
    
    # Visualize sample images
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Sample Images from Dataset', fontsize=16)
    
    # Show 4 eczema images
    eczema_indices = [i for i, label in enumerate(labels) if label == 1][:4]
    for i, idx in enumerate(eczema_indices):
        img = cv2.imread(image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title('Eczema')
        axes[0, i].axis('off')
    
    # Show 4 normal images
    normal_indices = [i for i, label in enumerate(labels) if label == 0][:4]
    for i, idx in enumerate(normal_indices):
        img = cv2.imread(image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].set_title('Normal')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    class_counts = [len(labels) - sum(labels), sum(labels)]
    colors = [config.CLASS_COLORS['Normal'], config.CLASS_COLORS['Eczema']]
    plt.pie(class_counts, labels=config.CLASS_NAMES, colors=colors, autopct='%1.1f%%')
    plt.title('Class Distribution')
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_dataset()
