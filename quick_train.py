#!/usr/bin/env python3
"""
Quick training script using only 100 images (50 per class)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import os
import time
from tqdm import tqdm
import config
from model import get_model
from data_preprocessing import create_data_loaders

class QuickTrainer:
    def __init__(self, model_type='resnet'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        self.model = get_model(model_type).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc="🏋️ Training")):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="✅ Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=15):
        best_val_acc = 0.0
        best_model_path = 'quick_model.pth'
        
        print(f"🚀 Starting training for {epochs} epochs...")
        print("="*60)
        
        for epoch in range(epochs):
            print(f'\n📅 Epoch {epoch + 1}/{epochs}')
            print('-' * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f'🏆 New best model saved! Val Acc: {val_acc:.2f}%')
        
        print(f'\n🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%')
        return best_model_path
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue', marker='o')
        ax1.plot(self.val_losses, label='Validation Loss', color='red', marker='o')
        ax1.set_title('📉 Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue', marker='o')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red', marker='o')
        ax2.set_title('📈 Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quick_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_loader, model_path='quick_model.pth'):
        """Evaluate model on test set"""
        print("\n🎯 Evaluating model on test set...")
        
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="🧪 Testing"):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = probs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        print('\n📊 Classification Report:')
        report = classification_report(all_targets, all_predictions, 
                                 target_names=config.CLASS_NAMES, output_dict=True)
        print(classification_report(all_targets, all_predictions, target_names=config.CLASS_NAMES))
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
        plt.title('🎯 Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('quick_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(all_targets, np.array(all_probabilities)[:, 1])
        auc_score = roc_auc_score(all_targets, np.array(all_probabilities)[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('📈 ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('quick_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }

def main():
    print("🏥 Quick Eczema Model Training (100 images only)")
    print("="*60)
    
    # Create data loaders with limited dataset
    print("📊 Creating data loaders with 50 images per class...")
    train_loader, val_loader, test_loader, data_splits = create_data_loaders(limit_per_class=50)
    
    print(f"✅ Data loaded successfully:")
    print(f"   🏋️ Training samples: {len(data_splits[0])}")
    print(f"   ✅ Validation samples: {len(data_splits[1])}")
    print(f"   🧪 Test samples: {len(data_splits[2])}")
    print(f"   📁 Total: {len(data_splits[0]) + len(data_splits[1]) + len(data_splits[2])} images")
    
    # Initialize trainer
    print("\n🤖 Initializing ResNet-50 model...")
    trainer = QuickTrainer(model_type='resnet')
    
    # Train model
    best_model_path = trainer.train(train_loader, val_loader, epochs=15)
    
    # Plot training history
    print("\n📊 Plotting training history...")
    trainer.plot_training_history()
    
    # Evaluate on test set
    results = trainer.evaluate_model(test_loader, best_model_path)
    
    print(f"\n🎉 Final Results:")
    print(f"   📈 Test Accuracy: {results['accuracy']:.3f}")
    print(f"   🎯 AUC Score: {results['auc_score']:.3f}")
    print(f"   💾 Model saved as: {best_model_path}")
    
    print("\n✅ Quick training completed! Model is ready for dashboard.")
    print("🌐 You can now run the dashboard and use classification features.")

if __name__ == "__main__":
    main()
