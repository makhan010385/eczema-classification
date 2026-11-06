import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config

class EczemaClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(EczemaClassifier, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modify the final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.NUM_CLASSES)
        )
        
        # Initialize weights for new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)

class CustomCNN(nn.Module):
    """Alternative custom CNN architecture"""
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # Calculate the input size for FC layer
        self._calculate_fc_input_size()
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, config.NUM_CLASSES)
    
    def _calculate_fc_input_size(self):
        # Create a dummy tensor to calculate the size
        dummy = torch.zeros(1, 3, *config.IMG_SIZE)
        dummy = self.pool(F.relu(self.bn1(self.conv1(dummy))))
        dummy = self.pool(F.relu(self.bn2(self.conv2(dummy))))
        dummy = self.pool(F.relu(self.bn3(self.conv3(dummy))))
        dummy = self.pool(F.relu(self.bn4(self.conv4(dummy))))
        self.fc_input_size = dummy.view(1, -1).size(1)
    
    def forward(self, x):
        # Convolutional layers with batch norm and ReLU
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def get_model(model_type='resnet'):
    """Get model instance"""
    if model_type == 'resnet':
        return EczemaClassifier(pretrained=True)
    elif model_type == 'custom':
        return CustomCNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    """Test model with dummy input"""
    model = get_model('resnet')
    dummy_input = torch.randn(1, 3, *config.IMG_SIZE)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Number of trainable parameters: {count_parameters(model)}")
    
    # Test custom CNN
    custom_model = get_model('custom')
    with torch.no_grad():
        custom_output = custom_model(dummy_input)
    
    print(f"Custom model output shape: {custom_output.shape}")
    print(f"Custom model parameters: {count_parameters(custom_model)}")

if __name__ == "__main__":
    test_model()
