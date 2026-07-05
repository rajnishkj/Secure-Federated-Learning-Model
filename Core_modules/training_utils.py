"""
Comprehensive Training Utilities for Federated Learning and Centralized Training

This module provides a complete suite of training utilities specifically designed for
both centralized and federated learning scenarios with the MHEALTH dataset. It includes
optimized training loops, advanced learning rate schedulers, evaluation metrics, and
visualization tools for comprehensive model training and analysis.

Key Features:
- Advanced learning rate schedulers with warmup and cosine annealing
- Federated averaging algorithms for distributed model aggregation
- Comprehensive evaluation metrics and confusion matrix visualization
- Optimized training loops with gradient clipping and memory management
- Model checkpointing and state management utilities
- Reproducibility utilities for consistent experimental results
- Device management and GPU optimization functions
- Professional visualization tools for training analysis

The module supports various optimizers (Adam, AdamW, SGD) and schedulers, providing
flexibility for different training scenarios while maintaining optimal performance
for sensor-based activity recognition tasks in federated learning environments.

Dependencies:
    torch: Core PyTorch framework for neural network training
    numpy: Numerical computations and array operations
    sklearn: Evaluation metrics and performance analysis
    matplotlib/seaborn: Professional visualization and plotting
    typing: Type hints for robust code documentation
    collections: OrderedDict for parameter management
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    """
    Configure comprehensive random seeding for reproducible experiments.
    
    This function sets all relevant random seeds to ensure deterministic behavior
    across PyTorch, NumPy, and CUDA operations. Essential for federated learning
    experiments where reproducibility is critical for fair comparisons.
    
    Args:
        seed (int, optional): Random seed value. Default: 42.
                            Use consistent values across experiment runs.
    """
    torch.manual_seed(seed)  # PyTorch CPU random number generation
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random number generation
    np.random.seed(seed)  # NumPy random number generation
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA operations
    torch.backends.cudnn.benchmark = False  # Disable CUDA optimization for reproducibility


def get_optimizer(model: nn.Module, optimizer_name: str = 'adam', **kwargs) -> optim.Optimizer:
    """
    Create optimizer for the model.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'adamw': optim.AdamW
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizers[optimizer_name](model.parameters(), **kwargs)


def get_lr_scheduler(optimizer: optim.Optimizer, scheduler_name: str = 'cosine', **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler ('cosine', 'step', 'exponential', 'warmup_cosine')
        **kwargs: Additional scheduler parameters
        
    Returns:
        Scheduler instance
    """
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == 'warmup_cosine':
        # Extract min_lr from kwargs if present
        min_lr = kwargs.pop('min_lr', 0.0)
        return WarmupCosineScheduler(optimizer, min_lr=min_lr, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay.

    Uses base learning rates for each parameter group to compute the current LR,
    avoiding compounding multiplication errors.
    """
    
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.num_warmup_steps = max(0, int(num_warmup_steps))
        self.num_training_steps = max(1, int(num_training_steps))
        self.min_lr = float(min_lr)
        self.current_step = 0
        # Store base learning rates
        self.base_lrs = [group.get('lr', 1e-3) for group in self.optimizer.param_groups]
        
    def _get_scale(self) -> float:
        if self.current_step < self.num_warmup_steps and self.num_warmup_steps > 0:
            return float(self.current_step) / float(self.num_warmup_steps)
        # Cosine decay over the remaining steps
        progress = float(self.current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    def step(self):
        """Advance one step and update learning rates."""
        self.current_step += 1
        scale = self._get_scale()
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = self.min_lr + (base_lr - self.min_lr) * scale
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'num_warmup_steps': self.num_warmup_steps,
            'num_training_steps': self.num_training_steps,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.num_warmup_steps = state_dict['num_warmup_steps']
        self.num_training_steps = state_dict['num_training_steps']
        self.min_lr = state_dict['min_lr']
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)


def train_epoch(model: nn.Module, train_loader, criterion: nn.Module, optimizer: optim.Optimizer, 
                device: torch.device, scheduler=None) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def evaluate_model(model: nn.Module, test_loader, criterion: nn.Module, 
                  device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on test data.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    # Calculate additional metrics
    precision = accuracy_score(all_targets, all_predictions)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'correct': correct,
        'total': total,
        'predictions': all_predictions,
        'targets': all_targets
    }


def federated_averaging(client_parameters: List[OrderedDict], client_sizes: List[int]) -> OrderedDict:
    """
    Aggregate parameters from clients using federated averaging.
    
    Args:
        client_parameters: List of client model parameters
        client_sizes: List of client dataset sizes
        
    Returns:
        Aggregated parameters
    """
    total_size = sum(client_sizes)
    global_parameters = OrderedDict()
    reference_params = client_parameters[0]
    
    # Initialize with zeros using float32 for accumulation
    for name, param in reference_params.items():
        global_parameters[name] = torch.zeros_like(param, dtype=torch.float32)
    
    # Weighted average with explicit casting to float
    for client_param, size in zip(client_parameters, client_sizes):
        weight = size / total_size
        for name, param in client_param.items():
            param_float = param.float()
            global_parameters[name] += (param_float * weight)
    
    # Convert back to original dtype after averaging
    for name, param in reference_params.items():
        if param.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]:
            global_parameters[name] = torch.round(global_parameters[name]).to(param.dtype)
        else:
            global_parameters[name] = global_parameters[name].to(param.dtype)
    
    return global_parameters


def add_noise_to_parameters(parameters: OrderedDict, noise_scale: float = 0.005) -> OrderedDict:
    """
    Add Gaussian noise to model parameters for differential privacy.
    
    Args:
        parameters: Model parameters
        noise_scale: Standard deviation of noise
        
    Returns:
        Parameters with added noise
    """
    noisy_parameters = OrderedDict()
    
    for name, param in parameters.items():
        # Skip adding noise to integer parameters
        if param.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]:
            noisy_parameters[name] = param.clone()
        else:
            noise = torch.randn_like(param) * noise_scale
            noisy_parameters[name] = param + noise
    
    return noisy_parameters


def add_noise_to_accuracy(accuracy: float, noise_scale: float = 0.02) -> float:
    """
    Add noise to accuracy metrics for privacy preservation.
    
    Args:
        accuracy: Original accuracy
        noise_scale: Noise scale factor
        
    Returns:
        Noisy accuracy
    """
    noise = np.random.normal(0, noise_scale * 100)
    noisy_accuracy = accuracy + noise
    return max(0, min(100, noisy_accuracy))


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_accuracy'], label='Train Accuracy', color='blue')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_model(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
               metrics: Dict[str, float], save_path: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        save_path: Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)


def load_model(model: nn.Module, optimizer: optim.Optimizer, load_path: str):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        load_path: Path to load the checkpoint from
        
    Returns:
        Dictionary with loaded information
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def get_device() -> torch.device:
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def print_model_info(model: nn.Module, input_size: Tuple[int, int] = (64, 23)):
    """
    Print model information.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    from models import count_parameters, get_model_summary
    
    summary = get_model_summary(model, input_size)
    
    print(f"Model: {summary['model_name']}")
    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Input size: {summary['input_size']}")
    print(f"Output size: {summary['output_size']}")
    print(f"Device: {next(model.parameters()).device}")
