"""
Neural Network Model Architectures for MHEALTH Activity Recognition

This module provides a comprehensive collection of neural network architectures
specifically designed and optimized for sensor-based human activity recognition
using the MHEALTH dataset. The models are tailored for processing multi-modal
sensor data from accelerometers, gyroscopes, and magnetometers.

Key Features:
- EfficientNet: Lightweight architecture with residual connections for optimal efficiency
- Conv1DNet: 1D convolutional network for temporal pattern recognition in sensor data
- TransformerNet: Self-attention based model for complex feature interactions
- Modular design supporting federated learning and differential privacy requirements
- Advanced initialization schemes and regularization techniques
- Flexible factory pattern for model instantiation and configuration

All models are optimized for:
- Multi-class activity classification (13 activity types)
- Real-time inference on mobile and edge devices
- Privacy-preserving training in federated learning scenarios
- Robust performance across heterogeneous data distributions

Dependencies:
    torch: Core PyTorch framework for neural network operations
    torch.nn: Neural network modules and layers
    torch.nn.functional: Functional interface for neural network operations
    math: Mathematical functions for positional encoding and initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EfficientNet(nn.Module):
    """
    EfficientNet Architecture for Sensor-Based Activity Recognition
    
    This implementation provides a lightweight yet powerful neural network architecture
    inspired by EfficientNet principles, specifically adapted for processing sensor
    data from mobile health devices. The model incorporates residual connections,
    batch normalization, and progressive dimension reduction for optimal efficiency.
    
    Key architectural features:
    - Progressive channel reduction for computational efficiency
    - Residual connections to combat vanishing gradients
    - Batch normalization for stable training dynamics
    - Dropout regularization for improved generalization
    - Xavier weight initialization for optimal convergence
    
    The model is designed to handle multi-modal sensor inputs including accelerometer,
    gyroscope, and magnetometer readings across three spatial axes, making it ideal
    for human activity recognition tasks in federated learning scenarios.
    
    Architecture Flow:
        Input (23 features) → FC(128) → Residual Block → FC(64) → FC(32) → Output (13 classes)
    """
    
    def __init__(self, input_dim=23, hidden_dim=128, num_classes=13, dropout_rate=0.3):
        """
        Initialize EfficientNet model with configurable architecture parameters.
        
        Args:
            input_dim (int, optional): Number of input sensor features. Default: 23.
                                     Typically 23 for MHEALTH (3 sensors × 3 axes + chest sensor).
            hidden_dim (int, optional): Base hidden layer dimension. Default: 128.
                                      Subsequent layers use progressive reduction (64, 32).
            num_classes (int, optional): Number of activity classification classes. Default: 13.
                                       Matches MHEALTH dataset activity categories.
            dropout_rate (float, optional): Dropout probability for regularization. Default: 0.3.
                                          Applied after each activation for overfitting prevention.
        """
        super(EfficientNet, self).__init__()
        
        # Progressive fully connected layers: Efficient dimension reduction architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Feature expansion layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Feature processing layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # First reduction layer
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)  # Second reduction layer
        self.fc_out = nn.Linear(hidden_dim // 4, num_classes)  # Classification output layer
        
        # Batch normalization layers: Stabilize training and improve convergence
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Residual connection module: Enable gradient flow and feature reuse
        self.residual1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Regularization layer: Prevent overfitting through random neuron deactivation
        self.dropout = nn.Dropout(dropout_rate)
        
        # Weight initialization: Apply Xavier initialization for optimal training dynamics
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer with residual connection
        identity = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + self.residual1(identity)  # Residual connection
        x = self.dropout(x)
        
        # Third layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fourth layer (no residual connection due to dimension mismatch)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc_out(x)
        
        return x


class Conv1DNet(nn.Module):
    """
    1D Convolutional Neural Network for time-series sensor data.
    Alternative architecture using 1D convolutions.
    """
    
    def __init__(self, input_dim=23, num_classes=13, dropout_rate=0.3):
        """
        Initialize Conv1DNet model.
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of activity classes
            dropout_rate (float): Dropout probability
        """
        super(Conv1DNet, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(64, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(32)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification layers
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Add channel dimension for 1D convolution
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)  # (batch_size, 32)
        
        # Classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerNet(nn.Module):
    """
    Transformer-based model for sensor data classification.
    Uses self-attention mechanism for feature learning.
    """
    
    def __init__(self, input_dim=23, num_classes=13, d_model=128, nhead=8, 
                 num_layers=2, dropout_rate=0.3):
        """
        Initialize TransformerNet model.
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of activity classes
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout_rate (float): Dropout probability
        """
        super(TransformerNet, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Input projection
        x = self.input_projection(x)  # (batch_size, input_dim, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, input_dim, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


def get_model(model_name='efficientnet', **kwargs):
    """
    Factory function to create model instances.
    
    Args:
        model_name (str): Name of the model to create
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Model instance
    """
    models = {
        'efficientnet': EfficientNet,
        'conv1d': Conv1DNet,
        'transformer': TransformerNet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, input_size=(64, 23)):
    """
    Get a summary of the model architecture and parameters.
    
    Args:
        model (nn.Module): The model to summarize
        input_size (tuple): Input tensor size (batch_size, features)
        
    Returns:
        dict: Model summary information
    """
    summary = {
        'model_name': model.__class__.__name__,
        'total_parameters': count_parameters(model),
        'input_size': input_size,
        'output_size': None
    }
    
    # Get output size by running a forward pass
    with torch.no_grad():
        dummy_input = torch.randn(*input_size)
        output = model(dummy_input)
        summary['output_size'] = output.shape
    
    return summary
