import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import config
from model import get_model
from data_preprocessing import get_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Page configuration
st.set_page_config(
    page_title="Eczema Classification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('resnet').to(device)
    
    # Load best model if exists
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device
    else:
        st.warning("No trained model found. Please train the model first.")
        return None, device

def preprocess_image(image):
    """Preprocess uploaded image"""
    # Convert PIL to numpy array
    image_array = np.array(image)
    
    # Apply transforms
    transform = get_transforms('test')
    transformed = transform(image=image_array)
    
    return transformed['image'], image_array

def predict_image(model, device, image_tensor):
    """Make prediction on image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        
        return predicted_class, confidence, probs[0].cpu().numpy()

def generate_grad_cam(model, device, image_tensor, target_class):
    """Generate Grad-CAM visualization"""
    try:
        # Define the target layer
        target_layers = [model.backbone.layer4[-1]]
        
        # Create GradCAM object
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Define targets
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(device), targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        return grayscale_cam
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None

def generate_lime_explanation(model, image_array, predict_fn):
    """Generate LIME explanation"""
    try:
        explainer = lime_image.LimeImageExplainer()
        
        # Resize image to reasonable size for LIME
        image_resized = cv2.resize(image_array, (224, 224))
        
        explanation = explainer.explain_instance(
            image_resized.astype('double'),
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=100
        )
        
        return explanation
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")
        return None

def model_predict_fn(images, model, device):
    """Prediction function for LIME"""
    model.eval()
    predictions = []
    
    for img in images:
        # Convert to tensor
        transform = get_transforms('test')
        transformed = transform(image=img)
        image_tensor = transformed['image']
        
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(device)
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            predictions.append(probs[0].cpu().numpy())
    
    return np.array(predictions)

def main():
    # Load model
    model, device = load_model()
    
    # Main header
    st.markdown('<h1 class="main-header">🏥 Eczema Classification System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Image Classification", "Model Analysis", "About"])
    
    if page == "Home":
        st.markdown("## Welcome to the Eczema Classification Dashboard")
        
        st.markdown("""
        ### 🎯 Project Overview
        This AI-powered system helps classify skin images as either **Normal** or **Eczema-infected** using deep learning.
        
        ### 🚀 Features
        - **Image Classification**: Upload skin images for instant classification
        - **XAI Explanations**: Understand model decisions using Grad-CAM and LIME
        - **Performance Metrics**: View detailed model performance analysis
        - **Interactive Dashboard**: User-friendly interface for clinical decision support
        
        ### 📊 Model Performance
        """)
        
        # Display model metrics if available
        if os.path.exists('best_model.pth'):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model", "ResNet-50")
            with col2:
                st.metric("Classes", "2 (Normal/Eczema)")
            with col3:
                st.metric("Input Size", "224x224")
            with col4:
                st.metric("Status", "✅ Trained")
        else:
            st.warning("⚠️ Model not trained yet. Please run training.py first.")
        
        st.markdown("### 📋 How to Use")
        
        st.markdown("""
        1. **Navigate to Image Classification** page
        2. **Upload** a skin image (JPG, PNG, etc.)
        3. **View** prediction results and confidence scores
        4. **Explore** XAI explanations to understand the decision
        5. **Analyze** model performance metrics
        """)
    
    elif page == "Image Classification":
        st.markdown("## 🖼️ Image Classification")
        
        if model is None:
            st.error("Model not loaded. Please train the model first.")
            return
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of skin for classification"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)
            
            # Preprocess and predict
            image_tensor, image_array = preprocess_image(image)
            predicted_class, confidence, probs = predict_image(model, device, image_tensor)
            
            with col2:
                st.markdown("### Prediction Results")
                
                # Display prediction
                class_name = config.CLASS_NAMES[predicted_class]
                color = config.CLASS_COLORS[class_name]
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {color};">Prediction: {class_name}</h3>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                fig = go.Figure(data=[
                    go.Bar(name='Probability', x=config.CLASS_NAMES, y=probs,
                          marker_color=[config.CLASS_COLORS[name] for name in config.CLASS_NAMES])
                ])
                fig.update_layout(title="Probability Distribution", yaxis_title="Probability")
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation based on prediction
                if predicted_class == 1:  # Eczema
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Medical Recommendation:</strong><br>
                        The model detects signs of eczema. Please consult a dermatologist for proper diagnosis and treatment.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Result:</strong><br>
                        The image appears to show normal skin. However, this is not a medical diagnosis. Consult a healthcare professional for any concerns.
                    </div>
                    """, unsafe_allow_html=True)
            
            # XAI Explanations
            st.markdown("### 🔍 Explainable AI (XAI) Analysis")
            
            tab1, tab2 = st.tabs(["Grad-CAM", "LIME"])
            
            with tab1:
                st.markdown("**Grad-CAM (Gradient-weighted Class Activation Mapping)**")
                st.markdown("Shows which regions of the image contributed most to the prediction.")
                
                # Generate Grad-CAM
                grad_cam = generate_grad_cam(model, device, image_tensor, predicted_class)
                
                if grad_cam is not None:
                    # Overlay Grad-CAM on original image
                    original_resized = cv2.resize(image_array, (224, 224))
                    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    superimposed = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_resized, caption="Original", use_container_width=True)
                    with col2:
                        st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)
            
            with tab2:
                st.markdown("**LIME (Local Interpretable Model-agnostic Explanations)**")
                st.markdown("Identifies the most important image regions for the prediction.")
                
                # Generate LIME explanation
                def predict_fn_lime(images):
                    return model_predict_fn(images, model, device)
                
                explanation = generate_lime_explanation(model, image_array, predict_fn_lime)
                
                if explanation is not None:
                    # Get explanation for the predicted class
                    temp, mask = explanation.get_image_and_mask(
                        predicted_class, 
                        positive_only=True, 
                        num_features=5, 
                        hide_rest=False
                    )
                    
                    # Create boundary visualization
                    img_boundry = mark_boundaries(temp/255.0, mask)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_array, caption="Original", use_container_width=True)
                    with col2:
                        st.image(img_boundry, caption="LIME Explanation", use_container_width=True)
    
    elif page == "Model Analysis":
        st.markdown("## 📊 Model Performance Analysis")
        
        # Check if training history exists
        if os.path.exists('training_history.png'):
            st.markdown("### Training History")
            st.image('training_history.png', use_container_width=True)
        else:
            st.info("Training history not available. Please train the model first.")
        
        # Check if confusion matrix exists
        if os.path.exists('confusion_matrix.png'):
            st.markdown("### Confusion Matrix")
            st.image('confusion_matrix.png', use_container_width=True)
        
        # Check if ROC curve exists
        if os.path.exists('roc_curve.png'):
            st.markdown("### ROC Curve")
            st.image('roc_curve.png', use_container_width=True)
        
        # Model architecture info
        st.markdown("### Model Architecture")
        
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Parameters", f"{total_params:,}")
            with col2:
                st.metric("Trainable Parameters", f"{trainable_params:,}")
            with col3:
                st.metric("Model Size", f"{total_params * 4 / 1024 / 1024:.2f} MB")
    
    elif page == "About":
        st.markdown("## ℹ️ About This Project")
        
        st.markdown("""
        ### 🎯 Purpose
        This project demonstrates the application of deep learning for medical image classification, specifically for detecting eczema from skin images.
        
        ### 🛠️ Technology Stack
        - **Deep Learning**: PyTorch, ResNet-50
        - **Computer Vision**: OpenCV, Albumentations
        - **XAI**: Grad-CAM, LIME, SHAP
        - **Dashboard**: Streamlit
        - **Visualization**: Matplotlib, Plotly, Seaborn
        
        ### 📁 Dataset
        The system is trained on a dataset containing:
        - Eczema-infected skin images
        - Normal skin images
        - Total: ~150 images (balanced dataset)
        
        ### ⚠️ Medical Disclaimer
        **This system is for educational and research purposes only and should not be used for medical diagnosis.**
        
        Always consult qualified healthcare professionals for medical concerns and treatment decisions.
        
        ### 🔬 Features
        - Binary classification (Normal vs Eczema)
        - Explainable AI visualizations
        - Real-time predictions
        - Performance metrics
        - Interactive dashboard
        
        ### 📈 Performance
        The model achieves high accuracy on the test dataset with robust performance metrics including:
        - Accuracy, Precision, Recall, F1-Score
        - ROC-AUC analysis
        - Confusion matrix visualization
        """)

if __name__ == "__main__":
    main()
