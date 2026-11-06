# 🏥 Eczema Classification System

An AI-powered system for classifying skin images as Normal or Eczema-infected using deep learning with comprehensive explainable AI (XAI) capabilities.

## 🎯 Features

- **Binary Classification**: Classifies skin images as Normal or Eczema
- **Deep Learning Model**: ResNet-50 based CNN architecture
- **Data Preprocessing**: Advanced augmentation and normalization techniques
- **XAI Explanations**: Grad-CAM, LIME, SHAP, and Counterfactual analysis
- **Interactive Dashboard**: Streamlit-based user interface
- **Comprehensive Analysis**: Performance metrics and visualizations
- **Real-time Predictions**: Upload and classify images instantly

## 📊 Dataset Structure

```
dataset/
├── Eczema/          # Infected eczema images
└── Normal/          # Normal skin images
```

## 🛠️ Installation

1. **Clone/Download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
python main.py --all
```
This will:
- Check dependencies and dataset
- Run data preprocessing and analysis
- Train the model
- Generate XAI explanations
- Create comprehensive analysis reports
- Launch the interactive dashboard

### Option 2: Step-by-Step Execution

1. **Check setup**:
   ```bash
   python main.py --check
   ```

2. **Preprocess data**:
   ```bash
   python main.py --preprocess
   ```

3. **Train model**:
   ```bash
   python main.py --train
   ```

4. **Run XAI analysis**:
   ```bash
   python main.py --xai
   ```

5. **Analyze results**:
   ```bash
   python main.py --analyze
   ```

6. **Launch dashboard**:
   ```bash
   python main.py --dashboard
   ```

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides:

- **Home**: Project overview and quick stats
- **Image Classification**: Upload images for instant classification with XAI explanations
- **Model Analysis**: View training history, confusion matrix, ROC curves
- **About**: Project information and medical disclaimer

Access at: `http://localhost:8501`

## 📁 Generated Files

After running the pipeline, you'll get:

- `best_model.pth` - Trained model weights
- `dataset_analysis.png` - Dataset visualization
- `training_history.png` - Training/validation curves
- `confusion_matrix.png` - Model confusion matrix
- `roc_curve.png` - ROC analysis
- `xai_analysis.png` - XAI explanations
- `comprehensive_analysis.png` - Complete performance analysis
- `interactive_dashboard.html` - Interactive analysis dashboard

## 🔬 XAI Features

### Grad-CAM
- Shows which image regions contributed most to the prediction
- Heatmap overlay on original image

### LIME (Local Interpretable Model-agnostic Explanations)
- Identifies important image regions
- Boundary visualization of influential areas

### SHAP (SHapley Additive exPlanations)
- Pixel-level importance scores
- Feature contribution analysis

### Counterfactual Analysis
- Generates minimal changes needed to flip prediction
- Shows what features would change the classification

## 📊 Model Architecture

- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Custom Layers**: Fully connected classifier with dropout
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Normal/Eczema)

## 🎨 Data Augmentation

- Horizontal/Vertical flips
- Random rotations
- Brightness/contrast adjustments
- Hue/saturation variations
- Gaussian noise and blur
- Normalization with ImageNet statistics

## 📈 Performance Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Class-wise performance
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **Confidence Analysis**: Prediction confidence distribution
- **Calibration**: Model confidence vs accuracy

## ⚠️ Medical Disclaimer

**This system is for educational and research purposes only and should not be used for medical diagnosis.**

Always consult qualified healthcare professionals for medical concerns and treatment decisions. The system's predictions are not a substitute for professional medical advice.

## 🛡️ Safety Considerations

- Model trained on limited dataset (~150 images)
- Not validated on diverse populations
- May have biases in training data
- Should be used as decision support tool only
- Requires clinical validation before deployment

## 🔧 Technical Stack

- **Deep Learning**: PyTorch, torchvision
- **Computer Vision**: OpenCV, Albumentations
- **XAI**: Grad-CAM, LIME, SHAP
- **Dashboard**: Streamlit
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Analysis**: scikit-learn, pandas, numpy

## 📋 Requirements

See `requirements.txt` for complete list of dependencies.

Key packages:
- torch>=2.0.0
- torchvision>=0.15.0
- streamlit>=1.28.0
- opencv-python>=4.8.0
- scikit-learn>=1.3.0
- shap>=0.42.0
- lime>=0.2.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with medical AI guidelines and regulations in your jurisdiction.

## 📞 Contact

For questions or issues, please create an issue in the repository.

---

**Remember**: This is a demonstration project. Always validate AI models thoroughly before any clinical use.
