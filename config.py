import os

# Dataset paths
DATASET_PATH = "dataset"
ECZEMA_PATH = os.path.join(DATASET_PATH, "Eczema")
NORMAL_PATH = os.path.join(DATASET_PATH, "Normal")

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model architecture
NUM_CLASSES = 2
CLASS_NAMES = ['Normal', 'Eczema']

# Colors for visualization
CLASS_COLORS = {
    'Normal': '#2E8B57',  # Sea green
    'Eczema': '#DC143C'   # Crimson
}

# XAI parameters
COUNTERFACTUAL_STEPS = 100
COUNTERFACTUAL_LR = 0.01
SHAP_SAMPLES = 50
