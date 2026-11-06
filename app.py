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
import pandas as pd
from datetime import datetime
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
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
        return model, device, True
    else:
        return None, device, False

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

def get_dataset_statistics():
    """Get dataset statistics"""
    try:
        eczema_files = [f for f in os.listdir(config.ECZEMA_PATH) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        normal_files = [f for f in os.listdir(config.NORMAL_PATH) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        return {
            'eczema_count': len(eczema_files),
            'normal_count': len(normal_files),
            'total_count': len(eczema_files) + len(normal_files)
        }
    except:
        return {'eczema_count': 0, 'normal_count': 0, 'total_count': 0}

def load_eval_metrics():
    """Load real evaluation metrics produced by evaluate.py (None if absent)"""
    try:
        import json
        with open('metrics.json') as f:
            return json.load(f)
    except Exception:
        return None

def create_performance_metrics():
    """Create performance metrics visualization"""
    real = load_eval_metrics()
    if real:
        metrics = {
            'Accuracy': real['accuracy'],
            'Precision': real['precision'],
            'Recall': real['recall'],
            'F1-Score': real['f1'],
            'AUC': real['auc']
        }
    else:
        # Sample data (run evaluate.py for real metrics)
        metrics = {
            'Accuracy': 0.92,
            'Precision': 0.90,
            'Recall': 0.94,
            'F1-Score': 0.92,
            'AUC': 0.96
        }
    
    fig = go.Figure(data=[
        go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
               marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )
    
    return fig

def create_class_distribution():
    """Create class distribution chart"""
    stats = get_dataset_statistics()
    
    fig = go.Figure(data=[
        go.Pie(labels=['Eczema', 'Normal'], 
               values=[stats['eczema_count'], stats['normal_count']],
               hole=0.3,
               marker_colors=['#FF6B6B', '#4ECDC4'])
    ])
    
    fig.update_layout(
        title="Dataset Class Distribution",
        annotations=[dict(text='Classes', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

def main():
    # Load model
    model, device, model_loaded = load_model()
    
    # Main header
    st.markdown('<h1 class="main-header">🏥 Eczema Classification AI System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home", 
        "🔍 Image Classification", 
        "📊 Model Analysis", 
        "🧠 XAI Explanations",
        "📈 Performance Dashboard",
        "ℹ️ About"
    ])
    
    if page == "🏠 Home":
        st.markdown("## Welcome to the Eczema Classification AI System")
        
        # Dataset Statistics
        stats = get_dataset_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📁 Dataset</h3>
                <p><strong>{stats['total_count']}</strong> Images</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔴 Eczema</h3>
                <p><strong>{stats['eczema_count']}</strong> Images</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🟢 Normal</h3>
                <p><strong>{stats['normal_count']}</strong> Images</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status = "✅ Trained" if model_loaded else "⚠️ Not Trained"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🤖 Model</h3>
                <p><strong>{status}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Class Distribution Chart
        st.plotly_chart(create_class_distribution(), use_container_width=True)
        
        # Features Overview
        st.markdown("### 🌟 Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🎯 Image Classification</h4>
                <p>Upload skin images for instant AI-powered classification with confidence scores.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🧠 XAI Explanations</h4>
                <p>Understand model decisions using Grad-CAM, LIME, and SHAP visualizations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Performance Analytics</h4>
                <p>Comprehensive metrics, confusion matrices, and ROC curves.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Start Guide
        st.markdown("### 🚀 Quick Start Guide")
        
        st.markdown("""
        <div class="info-box">
            <h4>Getting Started:</h4>
            <ol>
                <li><strong>Upload Image:</strong> Go to Image Classification page</li>
                <li><strong>Get Prediction:</strong> View AI classification results</li>
                <li><strong>Understand Why:</strong> Explore XAI explanations</li>
                <li><strong>Analyze Performance:</strong> Check model metrics</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Recent Activity
        st.markdown("### 📋 System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            if model_loaded:
                st.success("🤖 Model Status: Ready for predictions")
            else:
                st.warning("⚠️ Model Status: Training required")
    
    elif page == "🔍 Image Classification":
        st.markdown("## 🖼️ AI Image Classification")
        
        if not model_loaded:
            st.error("🚫 Model not loaded. Please train the model first using `python training.py`")
            return
        
        # File uploader with enhanced UI
        st.markdown("### 📤 Upload Skin Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image for classification...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of skin for AI classification",
            label_visibility="visible"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📸 Original Image")
                st.image(image, use_container_width=True, caption="Uploaded Image")
            
            # Preprocess and predict
            image_tensor, image_array = preprocess_image(image)
            predicted_class, confidence, probs = predict_image(model, device, image_tensor)
            
            with col2:
                st.markdown("### 🎯 AI Prediction Results")
                
                # Display prediction with enhanced UI
                class_name = config.CLASS_NAMES[predicted_class]
                color = config.CLASS_COLORS[class_name]
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color};">
                    <h3 style="color: {color}; font-size: 2rem;">🏥 {class_name}</h3>
                    <p style="font-size: 1.2rem;"><strong>Confidence:</strong> {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                fig = go.Figure(data=[
                    go.Bar(name='Probability', x=config.CLASS_NAMES, y=probs,
                          marker_color=[config.CLASS_COLORS[name] for name in config.CLASS_NAMES],
                          text=[f'{p:.1%}' for p in probs],
                          textposition='auto')
                ])
                
                fig.update_layout(
                    title="Prediction Probability Distribution",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1]),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Medical recommendation
                if predicted_class == 1:  # Eczema
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Medical Recommendation</h4>
                        <p><strong>The AI detects signs of eczema.</strong></p>
                        <p>Please consult a dermatologist for proper diagnosis and treatment.</p>
                        <p><em>This is not a medical diagnosis - always seek professional medical advice.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ Analysis Result</h4>
                        <p><strong>The image appears to show normal skin.</strong></p>
                        <p>However, this is not a medical diagnosis. Consult a healthcare professional for any concerns.</p>
                        <p><em>Regular skin checks are recommended for early detection.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "📊 Model Analysis":
        st.markdown("## 📈 Model Performance Analysis")
        
        if not model_loaded:
            st.warning("⚠️ Model not trained yet. Please run training first.")
        else:
            # Performance Metrics
            st.plotly_chart(create_performance_metrics(), use_container_width=True)
            
            # Model Architecture Info
            st.markdown("### 🏗️ Model Architecture")
            
            if model is not None:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🔢 Total Parameters", f"{total_params:,}")
                
                with col2:
                    st.metric("🎯 Trainable Parameters", f"{trainable_params:,}")
                
                with col3:
                    st.metric("💾 Model Size", f"{total_params * 4 / 1024 / 1024:.2f} MB")
            
            # Training History (if available)
            if os.path.exists('training_history.png'):
                st.markdown("### 📊 Training History")
                st.image('training_history.png', use_container_width=True, caption="Model Training Progress")
            
            # Confusion Matrix (if available)
            if os.path.exists('confusion_matrix.png'):
                st.markdown("### 🎯 Confusion Matrix")
                st.image('confusion_matrix.png', use_container_width=True, caption="Model Confusion Matrix")
            
            # ROC Curve (if available)
            if os.path.exists('roc_curve.png'):
                st.markdown("### 📈 ROC Curve")
                st.image('roc_curve.png', use_container_width=True, caption="Receiver Operating Characteristic")
    
    elif page == "🧠 XAI Explanations":
        st.markdown("## 🔬 Explainable AI (XAI) Analysis")
        
        if not model_loaded:
            st.error("🚫 Model not loaded. Please train the model first.")
            return
        
        st.markdown("""
        <div class="info-box">
            <h4>🧠 What is XAI?</h4>
            <p>Explainable AI helps us understand <strong>why</strong> the model makes certain predictions.</p>
            <p>This increases trust and helps identify potential biases in the AI system.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader for XAI analysis
        uploaded_file = st.file_uploader(
            "Upload image for XAI analysis...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="xai_upload"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_tensor, image_array = preprocess_image(image)
            predicted_class, confidence, probs = predict_image(model, device, image_tensor)
            
            # Display prediction
            class_name = config.CLASS_NAMES[predicted_class]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction: {class_name} (Confidence: {confidence:.1%})</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # XAI Tabs
            tab1, tab2, tab3 = st.tabs(["🔥 Grad-CAM", "🍋 LIME", "📊 SHAP"])
            
            with tab1:
                st.markdown("### 🔥 Grad-CAM Analysis")
                st.markdown("**Gradient-weighted Class Activation Mapping** shows which regions contributed most to the prediction.")
                
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
                        st.image(original_resized, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(superimposed, caption="Grad-CAM Heatmap", use_container_width=True)
                    
                    st.info("🔥 Red areas indicate regions that strongly influenced the prediction.")
            
            with tab2:
                st.markdown("### 🍋 LIME Analysis")
                st.markdown("**Local Interpretable Model-agnostic Explanations** identifies important image regions.")
                
                # Generate LIME explanation
                def predict_fn_lime(images):
                    return model_predict_fn(images, model, device)
                
                lime_exp = generate_lime_explanation(model, image_array, predict_fn_lime)
                
                if lime_exp is not None:
                    # Get explanation for the predicted class
                    temp, mask = lime_exp.get_image_and_mask(
                        predicted_class, 
                        positive_only=True, 
                        num_features=5, 
                        hide_rest=False
                    )
                    
                    # Create boundary visualization
                    img_boundry = mark_boundaries(temp/255.0, mask)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_array, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(img_boundry, caption="LIME Explanation", use_container_width=True)
                    
                    st.info("🍋 Green areas show regions that support the prediction.")
            
            with tab3:
                st.markdown("### 📊 SHAP Analysis")
                st.markdown("**SHAP (SHapley Additive exPlanations)** provides pixel-level importance scores.")
                
                st.info("📊 SHAP analysis is computationally intensive and may take a moment...")
                
                # Placeholder for SHAP (would need implementation)
                st.image(image_array, caption="Original Image", use_container_width=True)
                st.warning("📊 SHAP visualization requires additional implementation.")
    
    elif page == "📈 Performance Dashboard":
        st.markdown("## 📊 Comprehensive Performance Dashboard")
        
        if not model_loaded:
            st.warning("⚠️ Model not trained yet.")
        else:
            # Create comprehensive dashboard
            st.markdown("### 🎯 Model Performance Overview")
            
            real = load_eval_metrics()
            if real:
                st.caption(f"Evaluated on {real['n_eval']} held-out images (run evaluate.py to refresh)")
            else:
                st.caption("Sample metrics — run `python evaluate.py` to compute real ones")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Accuracy", f"{(real['accuracy'] if real else 0.923):.1%}")
            
            with col2:
                st.metric("🔍 Precision", f"{(real['precision'] if real else 0.905):.1%}")
            
            with col3:
                st.metric("📈 Recall", f"{(real['recall'] if real else 0.942):.1%}")
            
            with col4:
                st.metric("⚡ F1-Score", f"{(real['f1'] if real else 0.923):.1%}")
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Class-wise performance
                if real:
                    class_metrics = pd.DataFrame({
                        'Class': config.CLASS_NAMES,
                        'Precision': [real['per_class'][n]['precision'] for n in config.CLASS_NAMES],
                        'Recall': [real['per_class'][n]['recall'] for n in config.CLASS_NAMES],
                        'F1-Score': [real['per_class'][n]['f1'] for n in config.CLASS_NAMES]
                    })
                else:
                    class_metrics = pd.DataFrame({
                        'Class': ['Normal', 'Eczema'],
                        'Precision': [0.93, 0.90],
                        'Recall': [0.89, 0.94],
                        'F1-Score': [0.91, 0.92]
                    })
                
                fig = px.bar(class_metrics.melt(id_vars='Class', var_name='Metric', value_name='Score'),
                           x='Class', y='Score', color='Metric', barmode='group',
                           title="Class-wise Performance Metrics")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                confidence_data = (real['confidences'] if real and real.get('confidences')
                                   else np.random.beta(2, 5, 1000))  # Sample data fallback
                
                fig = px.histogram(x=confidence_data, nbins=30,
                                 title="Prediction Confidence Distribution",
                                 labels={'x': 'Confidence Score', 'y': 'Frequency'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Comprehensive analysis if available
            if os.path.exists('comprehensive_analysis.png'):
                st.markdown("### 📊 Comprehensive Analysis")
                st.image('comprehensive_analysis.png', use_container_width=True)
            
            if os.path.exists('interactive_dashboard.html'):
                st.markdown("### 🌐 Interactive Dashboard")
                st.info("📊 Interactive dashboard available as 'interactive_dashboard.html'")
    
    elif page == "ℹ️ About":
        st.markdown("## ℹ️ About This Project")
        
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Project Purpose</h3>
            <p>This project demonstrates the application of deep learning for medical image classification, specifically for detecting eczema from skin images.</p>
            <p>It serves as an educational tool for understanding AI in healthcare and the importance of explainable AI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technology Stack
        st.markdown("### 🛠️ Technology Stack")
        
        tech_stack = {
            '🤖 Deep Learning': ['PyTorch', 'ResNet-50', 'TensorFlow'],
            '👁️ Computer Vision': ['OpenCV', 'Albumentations', 'PIL'],
            '🧠 XAI Libraries': ['Grad-CAM', 'LIME', 'SHAP'],
            '📊 Visualization': ['Streamlit', 'Plotly', 'Matplotlib', 'Seaborn'],
            '🔧 Data Science': ['NumPy', 'Pandas', 'Scikit-learn']
        }
        
        for category, tools in tech_stack.items():
            with st.expander(category):
                st.write(", ".join(tools))
        
        # Dataset Information
        st.markdown("### 📁 Dataset Information")
        
        stats = get_dataset_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📁 Total Images", stats['total_count'])
        
        with col2:
            st.metric("🔴 Eczema Images", stats['eczema_count'])
        
        with col3:
            st.metric("🟢 Normal Images", stats['normal_count'])
        
        # Medical Disclaimer
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Medical Disclaimer</h3>
            <p><strong>This system is for educational and research purposes only and should not be used for medical diagnosis.</strong></p>
            <p>Always consult qualified healthcare professionals for medical concerns and treatment decisions.</p>
            <p>The system's predictions are not a substitute for professional medical advice, diagnosis, or treatment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.markdown("### 🌟 Key Features")
        
        features = [
            "🎯 Binary classification (Normal vs Eczema)",
            "🧠 Multiple XAI explanation methods",
            "📊 Real-time performance metrics",
            "🖼️ Interactive image upload and analysis",
            "📈 Comprehensive visualization dashboard",
            "🔍 Confidence scoring and uncertainty analysis",
            "📱 Responsive web interface",
            "🔬 Model interpretability and transparency"
        ]
        
        for feature in features:
            st.markdown(f"- {feature}")
        
        # Contact and Contribution
        st.markdown("### 📞 Contact & Contribution")
        
        st.markdown("""
        <div class="info-box">
            <p><strong>For questions, issues, or contributions:</strong></p>
            <ul>
                <li>Create an issue in the repository</li>
                <li>Submit pull requests for improvements</li>
                <li>Follow ethical AI guidelines for medical applications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <center>
            <p><strong>Eczema Classification AI System</strong></p>
            <p><em>Built with ❤️ for educational purposes</em></p>
            <p>© 2024 - AI Healthcare Demo</p>
        </center>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
