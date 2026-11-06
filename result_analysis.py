import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import config
from model import get_model
from data_preprocessing import create_data_loaders
import os

class ResultAnalyzer:
    def __init__(self, model_path='best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model('resnet').to(self.device)
        
        # Load trained model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")
    
    def evaluate_on_dataset(self, data_loader):
        """Evaluate model on entire dataset"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_features = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                _, predicted = probs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                
                # Extract features (before final layer)
                features = self.model.backbone.avgpool(self.model.backbone.layer4(self.model.backbone.layer3(
                    self.model.backbone.layer2(self.model.backbone.layer1(data)))))
                all_features.extend(features.view(features.size(0), -1).cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities),
            'features': np.array(all_features)
        }
    
    def generate_comprehensive_report(self, results):
        """Generate comprehensive analysis report"""
        predictions = results['predictions']
        targets = results['targets']
        probabilities = results['probabilities']
        
        # Basic metrics
        accuracy = np.mean(predictions == targets)
        report = classification_report(targets, predictions, target_names=config.CLASS_NAMES, output_dict=True)
        cm = confusion_matrix(targets, predictions)
        
        # ROC and AUC
        fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
        auc_score = roc_auc_score(targets, probabilities[:, 1])
        
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
        pr_auc = np.trapz(precision, recall)
        
        # Confidence analysis
        confidences = np.max(probabilities, axis=1)
        correct_predictions = (predictions == targets)
        
        # Create comprehensive visualization
        self.create_comprehensive_plots(results, accuracy, report, cm, fpr, tpr, auc_score, precision, recall, pr_auc, confidences, correct_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'confidences': confidences,
            'correct_predictions': correct_predictions
        }
    
    def create_comprehensive_plots(self, results, accuracy, report, cm, fpr, tpr, auc_score, precision, recall, pr_auc, confidences, correct_predictions):
        """Create comprehensive visualization plots"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Confusion Matrix
        plt.subplot(3, 4, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. ROC Curve
        plt.subplot(3, 4, 2)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # 3. Precision-Recall Curve
        plt.subplot(3, 4, 3)
        plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        # 4. Class-wise Performance
        plt.subplot(3, 4, 4)
        classes = list(report.keys())[:-2]  # Exclude 'accuracy' and 'macro avg'
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [report[c][metric] for c in classes]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Class-wise Performance')
        plt.xticks(x + width, classes)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Confidence Distribution
        plt.subplot(3, 4, 5)
        plt.hist(confidences[correct_predictions], bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(confidences[~correct_predictions], bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Probability Distribution by Class
        plt.subplot(3, 4, 6)
        for i, class_name in enumerate(config.CLASS_NAMES):
            class_probs = results['probabilities'][results['targets'] == i, i]
            plt.hist(class_probs, bins=20, alpha=0.7, label=class_name, color=config.CLASS_COLORS[class_name])
        
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution by True Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Feature Distribution (t-SNE visualization)
        plt.subplot(3, 4, 7)
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(results['features'])
            
            for i, class_name in enumerate(config.CLASS_NAMES):
                mask = results['targets'] == i
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           label=class_name, alpha=0.7, color=config.CLASS_COLORS[class_name])
            
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.title('Feature Space Visualization')
            plt.legend()
            plt.grid(True, alpha=0.3)
        except ImportError:
            plt.text(0.5, 0.5, 't-SNE\nnot available', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 8. Calibration Curve
        plt.subplot(3, 4, 8)
        try:
            from sklearn.calibration import calibration_curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                results['targets'], results['probabilities'][:, 1], n_bins=10)
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.title("Calibration Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)
        except ImportError:
            plt.text(0.5, 0.5, 'Calibration curve\nnot available', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 9. Performance Summary Table
        plt.subplot(3, 4, 9)
        plt.axis('off')
        
        # Create summary text
        summary_text = f"""
        Performance Summary
        ─────────────────────
        Overall Accuracy: {accuracy:.3f}
        AUC Score: {auc_score:.3f}
        PR AUC: {pr_auc:.3f}
        
        Normal Class:
        Precision: {report['Normal']['precision']:.3f}
        Recall: {report['Normal']['recall']:.3f}
        F1-Score: {report['Normal']['f1-score']:.3f}
        
        Eczema Class:
        Precision: {report['Eczema']['precision']:.3f}
        Recall: {report['Eczema']['recall']:.3f}
        F1-Score: {report['Eczema']['f1-score']:.3f}
        
        Macro Average:
        Precision: {report['macro avg']['precision']:.3f}
        Recall: {report['macro avg']['recall']:.3f}
        F1-Score: {report['macro avg']['f1-score']:.3f}
        """
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 10. Error Analysis
        plt.subplot(3, 4, 10)
        false_positives = (results['predictions'] == 1) & (results['targets'] == 0)
        false_negatives = (results['predictions'] == 0) & (results['targets'] == 1)
        
        error_types = ['True Pos', 'True Neg', 'False Pos', 'False Neg']
        error_counts = [
            np.sum((results['predictions'] == 1) & (results['targets'] == 1)),
            np.sum((results['predictions'] == 0) & (results['targets'] == 0)),
            np.sum(false_positives),
            np.sum(false_negatives)
        ]
        colors = ['green', 'blue', 'orange', 'red']
        
        plt.pie(error_counts, labels=error_types, colors=colors, autopct='%1.1f%%')
        plt.title('Prediction Types')
        
        # 11. Confidence vs Accuracy
        plt.subplot(3, 4, 11)
        confidence_bins = np.linspace(0, 1, 11)
        accuracy_by_confidence = []
        
        for i in range(len(confidence_bins)-1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if np.sum(mask) > 0:
                acc = np.mean(correct_predictions[mask])
                accuracy_by_confidence.append(acc)
            else:
                accuracy_by_confidence.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.plot(bin_centers, accuracy_by_confidence, 'bo-')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        plt.xlabel('Confidence Bin')
        plt.ylabel('Accuracy')
        plt.title('Confidence vs Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 12. Class Imbalance Impact
        plt.subplot(3, 4, 12)
        class_counts = np.bincount(results['targets'])
        plt.bar(config.CLASS_NAMES, class_counts, color=[config.CLASS_COLORS[name] for name in config.CLASS_NAMES])
        plt.ylabel('Count')
        plt.title('Test Set Class Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comprehensive analysis saved as 'comprehensive_analysis.png'")
    
    def generate_interactive_dashboard(self, results):
        """Generate interactive dashboard using Plotly"""
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve', 
                          'Class Performance', 'Confidence Distribution', 'Feature Space'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Confusion Matrix
        cm = confusion_matrix(results['targets'], results['predictions'])
        fig.add_trace(
            go.Heatmap(z=cm, x=config.CLASS_NAMES, y=config.CLASS_NAMES, 
                      colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(results['targets'], results['probabilities'][:, 1])
        auc_score = roc_auc_score(results['targets'], results['probabilities'][:, 1])
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc_score:.3f})'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')),
            row=1, col=2
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(results['targets'], results['probabilities'][:, 1])
        pr_auc = np.trapz(precision, recall)
        
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC = {pr_auc:.3f})'),
            row=2, col=1
        )
        
        # Class Performance
        report = classification_report(results['targets'], results['predictions'], 
                                     target_names=config.CLASS_NAMES, output_dict=True)
        classes = list(report.keys())[:-2]
        metrics = ['precision', 'recall', 'f1-score']
        
        for metric in metrics:
            values = [report[c][metric] for c in classes]
            fig.add_trace(
                go.Bar(x=classes, y=values, name=metric),
                row=2, col=2
            )
        
        # Confidence Distribution
        confidences = np.max(results['probabilities'], axis=1)
        correct_predictions = (results['predictions'] == results['targets'])
        
        fig.add_trace(
            go.Histogram(x=confidences[correct_predictions], name='Correct', opacity=0.7),
            row=3, col=1
        )
        fig.add_trace(
            go.Histogram(x=confidences[~correct_predictions], name='Incorrect', opacity=0.7),
            row=3, col=1
        )
        
        # Feature Space (if available)
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(results['features'])
            
            for i, class_name in enumerate(config.CLASS_NAMES):
                mask = results['targets'] == i
                fig.add_trace(
                    go.Scatter(x=features_2d[mask, 0], y=features_2d[mask, 1], 
                              mode='markers', name=class_name, opacity=0.7),
                    row=3, col=2
                )
        except ImportError:
            pass
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Model Analysis Dashboard",
            showlegend=True,
            height=1200
        )
        
        # Save as HTML
        fig.write_html("interactive_dashboard.html")
        print("Interactive dashboard saved as 'interactive_dashboard.html'")
        
        return fig

def main():
    """Main function to run result analysis"""
    # Create data loaders
    _, _, test_loader, _ = create_data_loaders()
    
    # Initialize analyzer
    analyzer = ResultAnalyzer()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    results = analyzer.evaluate_on_dataset(test_loader)
    
    # Generate comprehensive report
    print("Generating comprehensive analysis...")
    metrics = analyzer.generate_comprehensive_report(results)
    
    # Generate interactive dashboard
    print("Generating interactive dashboard...")
    analyzer.generate_interactive_dashboard(results)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Test Accuracy: {metrics['accuracy']:.3f}")
    print(f"AUC Score: {metrics['auc_score']:.3f}")
    print(f"PR AUC: {metrics['pr_auc']:.3f}")
    
    print("\nClass-wise Performance:")
    for class_name in config.CLASS_NAMES:
        print(f"{class_name}:")
        print(f"  Precision: {metrics['classification_report'][class_name]['precision']:.3f}")
        print(f"  Recall: {metrics['classification_report'][class_name]['recall']:.3f}")
        print(f"  F1-Score: {metrics['classification_report'][class_name]['f1-score']:.3f}")
    
    print("\nFiles generated:")
    print("- comprehensive_analysis.png")
    print("- interactive_dashboard.html")

if __name__ == "__main__":
    main()
