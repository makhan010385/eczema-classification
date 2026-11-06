import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import config
from model import get_model
from data_preprocessing import get_transforms
import os

class XAIAnalyzer:
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
    
    def preprocess_image(self, image_path):
        """Preprocess image for analysis"""
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transform = get_transforms('test')
        transformed = transform(image=image_rgb)
        
        return transformed['image'], image_rgb
    
    def generate_grad_cam(self, image_tensor, target_class):
        """Generate Grad-CAM visualization"""
        try:
            # Define target layer
            target_layers = [self.model.backbone.layer4[-1]]
            
            # Create GradCAM object
            cam = GradCAM(model=self.model, target_layers=target_layers)
            
            # Define targets
            targets = [ClassifierOutputTarget(target_class)]
            
            # Generate CAM
            grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(self.device), targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            return grayscale_cam
        except Exception as e:
            print(f"Error generating Grad-CAM: {str(e)}")
            return None
    
    def generate_lime_explanation(self, image_array, target_class):
        """Generate LIME explanation"""
        try:
            explainer = lime_image.LimeImageExplainer()
            
            # Resize image if needed
            image_resized = cv2.resize(image_array, (224, 224))
            
            def predict_fn(images):
                """Prediction function for LIME"""
                predictions = []
                for img in images:
                    transform = get_transforms('test')
                    transformed = transform(image=img)
                    image_tensor = transformed['image']
                    
                    with torch.no_grad():
                        image_tensor = image_tensor.unsqueeze(0).to(self.device)
                        outputs = self.model(image_tensor)
                        probs = F.softmax(outputs, dim=1)
                        predictions.append(probs[0].cpu().numpy())
                
                return np.array(predictions)
            
            explanation = explainer.explain_instance(
                image_resized.astype('double'),
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=100
            )
            
            return explanation
        except Exception as e:
            print(f"Error generating LIME explanation: {str(e)}")
            return None
    
    def generate_shap_explanation(self, image_tensor, image_array):
        """Generate SHAP explanation"""
        try:
            # Create SHAP explainer
            background = torch.zeros((1, 3, *config.IMG_SIZE)).to(self.device)
            
            def model_fn(x):
                with torch.no_grad():
                    outputs = self.model(x)
                    return F.softmax(outputs, dim=1)
            
            explainer = shap.GradientExplainer(model_fn, background)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(image_tensor.unsqueeze(0).to(self.device))
            
            return shap_values
        except Exception as e:
            print(f"Error generating SHAP explanation: {str(e)}")
            return None
    
    def generate_counterfactual(self, image_tensor, target_class, steps=100, lr=0.01):
        """Generate counterfactual explanation"""
        try:
            # Create a copy of the input tensor for optimization
            counterfactual = image_tensor.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([counterfactual], lr=lr)
            
            # Target class (opposite of predicted)
            target_label = 1 - target_class
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(counterfactual.unsqueeze(0))
                loss = F.cross_entropy(outputs, torch.tensor([target_label]).to(self.device))
                
                # Add L2 regularization to keep image realistic
                l2_loss = torch.mean((counterfactual - image_tensor) ** 2)
                total_loss = loss + 0.01 * l2_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Clamp values to valid range
                counterfactual.data = torch.clamp(counterfactual.data, 0, 1)
                
                if step % 20 == 0:
                    print(f"Step {step}, Loss: {total_loss.item():.4f}")
            
            return counterfactual.detach()
        except Exception as e:
            print(f"Error generating counterfactual: {str(e)}")
            return None
    
    def analyze_image(self, image_path, save_results=True):
        """Comprehensive XAI analysis of an image"""
        print(f"Analyzing image: {image_path}")
        
        # Preprocess image
        image_tensor, image_rgb = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor.unsqueeze(0).to(self.device))
            probs = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
        
        print(f"Prediction: {config.CLASS_NAMES[predicted_class]} (Confidence: {confidence:.2%})")
        
        # Generate explanations
        results = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy()
        }
        
        # Grad-CAM
        print("Generating Grad-CAM...")
        grad_cam = self.generate_grad_cam(image_tensor, predicted_class)
        if grad_cam is not None:
            results['grad_cam'] = grad_cam
        
        # LIME
        print("Generating LIME explanation...")
        lime_exp = self.generate_lime_explanation(image_rgb, predicted_class)
        if lime_exp is not None:
            results['lime'] = lime_exp
        
        # SHAP
        print("Generating SHAP explanation...")
        shap_vals = self.generate_shap_explanation(image_tensor, image_rgb)
        if shap_vals is not None:
            results['shap'] = shap_vals
        
        # Counterfactual
        print("Generating counterfactual...")
        counterfactual = self.generate_counterfactual(image_tensor, predicted_class)
        if counterfactual is not None:
            results['counterfactual'] = counterfactual
        
        # Save visualizations
        if save_results:
            self.save_visualizations(results, image_rgb)
        
        return results
    
    def save_visualizations(self, results, original_image):
        """Save all visualization results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'XAI Analysis - Predicted: {config.CLASS_NAMES[results["predicted_class"]]}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Probability distribution
        probs = results['probabilities']
        axes[0, 1].bar(config.CLASS_NAMES, probs, color=[config.CLASS_COLORS[name] for name in config.CLASS_NAMES])
        axes[0, 1].set_title('Probability Distribution')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].set_ylim(0, 1)
        
        # Grad-CAM
        if 'grad_cam' in results:
            grad_cam = results['grad_cam']
            original_resized = cv2.resize(original_image, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            superimposed = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)
            axes[0, 2].imshow(superimposed)
            axes[0, 2].set_title('Grad-CAM Heatmap')
            axes[0, 2].axis('off')
        else:
            axes[0, 2].text(0.5, 0.5, 'Grad-CAM\nFailed', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
        
        # LIME
        if 'lime' in results:
            lime_exp = results['lime']
            temp, mask = lime_exp.get_image_and_mask(
                results['predicted_class'], 
                positive_only=True, 
                num_features=5, 
                hide_rest=False
            )
            img_boundry = mark_boundaries(temp/255.0, mask)
            axes[1, 0].imshow(img_boundry)
            axes[1, 0].set_title('LIME Explanation')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'LIME\nFailed', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # SHAP
        if 'shap' in results:
            shap_vals = results['shap']
            # Take the first class SHAP values
            shap_image = np.abs(shap_vals[0][0]).mean(axis=0)
            axes[1, 1].imshow(shap_image, cmap='hot')
            axes[1, 1].set_title('SHAP Values')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'SHAP\nFailed', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
        
        # Counterfactual
        if 'counterfactual' in results:
            counterfactual = results['counterfactual'].cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            counterfactual = np.clip(counterfactual * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
            axes[1, 2].imshow(counterfactual)
            axes[1, 2].set_title(f'Counterfactual\n({config.CLASS_NAMES[1-results["predicted_class"]]})')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, 'Counterfactual\nFailed', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('xai_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("XAI analysis saved as 'xai_analysis.png'")

def main():
    """Main function to run XAI analysis"""
    # Initialize analyzer
    analyzer = XAIAnalyzer()
    
    # Analyze a sample image (you can change this path)
    sample_images = []
    
    # Get sample eczema image
    eczema_files = [f for f in os.listdir(config.ECZEMA_PATH) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if eczema_files:
        sample_images.append(os.path.join(config.ECZEMA_PATH, eczema_files[0]))
    
    # Get sample normal image
    normal_files = [f for f in os.listdir(config.NORMAL_PATH) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if normal_files:
        sample_images.append(os.path.join(config.NORMAL_PATH, normal_files[0]))
    
    # Analyze sample images
    for image_path in sample_images:
        print(f"\n{'='*50}")
        results = analyzer.analyze_image(image_path)
        print(f"Analysis completed for: {image_path}")

if __name__ == "__main__":
    main()
