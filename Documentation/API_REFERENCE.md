# API Reference

## Overview

This document provides comprehensive API documentation for all modules in the Privacy-Preserving Machine Learning for IoT Data project. Each module is designed with modularity and extensibility in mind, providing clear interfaces for research and development.

## Module Index

- [Dataset Loader](#dataset-loader-dataset_loaderpy)
- [Models](#models-modelspy)
- [Training Utilities](#training-utilities-training_utilspy)
- [Differential Privacy Utilities](#differential-privacy-utilities-dp_utilspy)
- [Data Partitioning](#data-partitioning-partitioningpy)

---

## Dataset Loader (`dataset_loader.py`)

### Classes

#### `MHEALTHDataset`

PyTorch Dataset class for loading and preprocessing MHEALTH sensor data.

```python
class MHEALTHDataset(Dataset):
    def __init__(self, data_path, transform=None, standardize=True)
```

**Parameters:**
- `data_path` (str): Path to the MHEALTH dataset directory
- `transform` (callable, optional): Optional transform to be applied on samples
- `standardize` (bool): Whether to standardize features using Z-score normalization

**Attributes:**
- `data` (np.ndarray): Preprocessed sensor data
- `labels` (np.ndarray): Activity labels
- `transform` (callable): Data transformation function

**Methods:**

##### `__len__()`
Returns the total number of samples in the dataset.

**Returns:**
- `int`: Number of samples

##### `__getitem__(idx)`
Retrieves a sample and its label by index.

**Parameters:**
- `idx` (int): Sample index

**Returns:**
- `tuple`: (sample, label) as PyTorch tensors

**Example:**
```python
from dataset_loader import MHEALTHDataset

# Load dataset
dataset = MHEALTHDataset('MHEALTHDATASET/', standardize=True)

# Get dataset info
print(f"Dataset size: {len(dataset)}")
print(f"Feature dimension: {dataset.data.shape[1]}")
print(f"Number of classes: {len(np.unique(dataset.labels))}")

# Get a sample
sample, label = dataset[0]
```

### Functions

#### `create_data_loaders()`

Creates train and test data loaders for centralized training.

```python
def create_data_loaders(data_path, batch_size=64, test_size=0.2, 
                       random_state=42, num_workers=0)
```

**Parameters:**
- `data_path` (str): Path to the MHEALTH dataset
- `batch_size` (int): Batch size for data loaders
- `test_size` (float): Fraction of data for testing (0.0-1.0)
- `random_state` (int): Random seed for reproducibility
- `num_workers` (int): Number of worker processes for data loading

**Returns:**
- `tuple`: (train_loader, test_loader, dataset_info)
  - `train_loader` (DataLoader): Training data loader
  - `test_loader` (DataLoader): Testing data loader
  - `dataset_info` (dict): Dataset metadata

**Example:**
```python
train_loader, test_loader, info = create_data_loaders(
    data_path='MHEALTHDATASET/',
    batch_size=512,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {info['train_samples']}")
print(f"Test samples: {info['test_samples']}")
print(f"Features: {info['num_features']}")
print(f"Classes: {info['num_classes']}")
```

#### `create_federated_data_loaders()`

Creates federated data loaders with non-IID partitioning.

```python
def create_federated_data_loaders(data_path, num_clients=5, batch_size=64, 
                                 alpha=0.5, random_state=42)
```

**Parameters:**
- `data_path` (str): Path to the MHEALTH dataset
- `num_clients` (int): Number of federated clients
- `batch_size` (int): Batch size for data loaders
- `alpha` (float): Dirichlet distribution parameter (lower = more heterogeneous)
- `random_state` (int): Random seed for reproducibility

**Returns:**
- `list`: List of (train_loader, test_loader) tuples for each client

**Example:**
```python
client_loaders = create_federated_data_loaders(
    data_path='MHEALTHDATASET/',
    num_clients=5,
    batch_size=256,
    alpha=0.5,
    random_state=42
)

# Access individual client data
for i, (train_loader, test_loader) in enumerate(client_loaders):
    print(f"Client {i+1}: {len(train_loader.dataset)} training samples")
```

---

## Models (`models.py`)

### Classes

#### `EfficientNet`

Optimized neural network architecture for sensor data classification.

```python
class EfficientNet(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=128, num_classes=13, dropout_rate=0.3)
```

**Parameters:**
- `input_dim` (int): Number of input features
- `hidden_dim` (int): Hidden layer dimension
- `num_classes` (int): Number of output classes
- `dropout_rate` (float): Dropout probability for regularization

**Architecture:**
- Input layer: Linear transformation with batch normalization
- Hidden layers: 4 fully connected layers with residual connections
- Output layer: Final classification layer
- Regularization: Dropout and batch normalization throughout

**Methods:**

##### `forward(x)`
Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch_size, input_dim)

**Returns:**
- `torch.Tensor`: Output logits of shape (batch_size, num_classes)

**Example:**
```python
from models import EfficientNet

# Create model
model = EfficientNet(
    input_dim=23,
    hidden_dim=256,
    num_classes=13,
    dropout_rate=0.2
)

# Forward pass
import torch
x = torch.randn(32, 23)  # Batch of 32 samples
output = model(x)  # Shape: (32, 13)
```

#### `Conv1DNet`

1D Convolutional neural network for time-series sensor data.

```python
class Conv1DNet(nn.Module):
    def __init__(self, input_dim=23, num_classes=13, dropout_rate=0.3)
```

**Parameters:**
- `input_dim` (int): Number of input features
- `num_classes` (int): Number of output classes
- `dropout_rate` (float): Dropout probability

**Architecture:**
- 1D convolutions with increasing then decreasing channel dimensions
- Global average pooling for feature aggregation
- Final classification layers

#### `TransformerNet`

Transformer-based architecture with self-attention mechanism.

```python
class TransformerNet(nn.Module):
    def __init__(self, input_dim=23, num_classes=13, d_model=128, 
                 nhead=8, num_layers=2, dropout_rate=0.3)
```

**Parameters:**
- `input_dim` (int): Number of input features
- `num_classes` (int): Number of output classes
- `d_model` (int): Model dimension for transformer
- `nhead` (int): Number of attention heads
- `num_layers` (int): Number of transformer encoder layers
- `dropout_rate` (float): Dropout probability

### Functions

#### `get_model()`

Factory function for creating model instances.

```python
def get_model(model_name='efficientnet', **kwargs)
```

**Parameters:**
- `model_name` (str): Model architecture name ('efficientnet', 'conv1d', 'transformer')
- `**kwargs`: Additional arguments passed to model constructor

**Returns:**
- `nn.Module`: Instantiated model

**Example:**
```python
from models import get_model

# Create EfficientNet model
model = get_model(
    model_name='efficientnet',
    input_dim=23,
    hidden_dim=256,
    num_classes=13,
    dropout_rate=0.2
)

# Create Conv1D model
conv_model = get_model(
    model_name='conv1d',
    input_dim=23,
    num_classes=13
)
```

#### `count_parameters()`

Count the number of trainable parameters in a model.

```python
def count_parameters(model)
```

**Parameters:**
- `model` (nn.Module): PyTorch model

**Returns:**
- `int`: Number of trainable parameters

**Example:**
```python
from models import get_model, count_parameters

model = get_model('efficientnet', hidden_dim=256)
param_count = count_parameters(model)
print(f"Model has {param_count:,} trainable parameters")
```

#### `get_model_summary()`

Get comprehensive model information and statistics.

```python
def get_model_summary(model, input_size=(64, 23))
```

**Parameters:**
- `model` (nn.Module): PyTorch model to analyze
- `input_size` (tuple): Input tensor dimensions for analysis

**Returns:**
- `dict`: Model summary with architecture details

**Example:**
```python
from models import get_model, get_model_summary

model = get_model('efficientnet')
summary = get_model_summary(model, input_size=(32, 23))

print(f"Model: {summary['model_name']}")
print(f"Parameters: {summary['total_parameters']:,}")
print(f"Input size: {summary['input_size']}")
print(f"Output size: {summary['output_size']}")
```

---

## Training Utilities (`training_utils.py`)

### Functions

#### `set_seed()`

Set random seeds for reproducible experiments.

```python
def set_seed(seed=42)
```

**Parameters:**
- `seed` (int): Random seed value

**Example:**
```python
from training_utils import set_seed

set_seed(42)  # Ensures reproducible results
```

#### `get_optimizer()`

Create and configure optimizers for model training.

```python
def get_optimizer(model, optimizer_name='adam', **kwargs)
```

**Parameters:**
- `model` (nn.Module): PyTorch model
- `optimizer_name` (str): Optimizer type ('adam', 'sgd', 'adamw')
- `**kwargs`: Optimizer-specific parameters

**Returns:**
- `torch.optim.Optimizer`: Configured optimizer

**Example:**
```python
from training_utils import get_optimizer
from models import get_model

model = get_model('efficientnet')

# Create AdamW optimizer
optimizer = get_optimizer(
    model,
    optimizer_name='adamw',
    lr=0.001,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

#### `get_lr_scheduler()`

Create learning rate schedulers for training optimization.

```python
def get_lr_scheduler(optimizer, scheduler_name='cosine', **kwargs)
```

**Parameters:**
- `optimizer` (torch.optim.Optimizer): Optimizer instance
- `scheduler_name` (str): Scheduler type ('cosine', 'step', 'exponential', 'warmup_cosine')
- `**kwargs`: Scheduler-specific parameters

**Returns:**
- Learning rate scheduler instance

**Example:**
```python
from training_utils import get_optimizer, get_lr_scheduler

# Create optimizer and scheduler
optimizer = get_optimizer(model, 'adamw', lr=0.001)
scheduler = get_lr_scheduler(
    optimizer,
    scheduler_name='warmup_cosine',
    num_warmup_steps=100,
    num_training_steps=1000,
    min_lr=1e-6
)
```

#### `train_epoch()`

Train model for one epoch with comprehensive metrics tracking.

```python
def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None)
```

**Parameters:**
- `model` (nn.Module): PyTorch model to train
- `train_loader` (DataLoader): Training data loader
- `criterion` (nn.Module): Loss function
- `optimizer` (torch.optim.Optimizer): Optimizer for parameter updates
- `device` (torch.device): Training device (CPU/GPU)
- `scheduler` (optional): Learning rate scheduler

**Returns:**
- `dict`: Training metrics including loss, accuracy, and counts

**Example:**
```python
from training_utils import train_epoch, get_optimizer, get_device
import torch.nn as nn

# Setup training components
model = get_model('efficientnet').to(get_device())
optimizer = get_optimizer(model, 'adamw', lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train for one epoch
metrics = train_epoch(
    model, train_loader, criterion, optimizer, get_device()
)

print(f"Training loss: {metrics['loss']:.4f}")
print(f"Training accuracy: {metrics['accuracy']:.2f}%")
```

#### `evaluate_model()`

Evaluate model performance on test data with detailed metrics.

```python
def evaluate_model(model, test_loader, criterion, device)
```

**Parameters:**
- `model` (nn.Module): PyTorch model to evaluate
- `test_loader` (DataLoader): Test data loader
- `criterion` (nn.Module): Loss function
- `device` (torch.device): Evaluation device

**Returns:**
- `dict`: Evaluation metrics including predictions and targets

**Example:**
```python
from training_utils import evaluate_model

# Evaluate model
eval_metrics = evaluate_model(model, test_loader, criterion, device)

print(f"Test accuracy: {eval_metrics['accuracy']:.2f}%")
print(f"Test loss: {eval_metrics['loss']:.4f}")

# Access predictions for further analysis
predictions = eval_metrics['predictions']
targets = eval_metrics['targets']
```

#### `federated_averaging()`

Aggregate client parameters using FedAvg algorithm.

```python
def federated_averaging(client_parameters, client_sizes)
```

**Parameters:**
- `client_parameters` (List[OrderedDict]): List of client model parameters
- `client_sizes` (List[int]): List of client dataset sizes for weighting

**Returns:**
- `OrderedDict`: Aggregated global model parameters

**Example:**
```python
from training_utils import federated_averaging
import copy

# Simulate client training (normally done on separate devices)
client_params = []
client_sizes = []

for client_id, (train_loader, _) in enumerate(client_loaders):
    # Clone global model for client
    client_model = copy.deepcopy(global_model)
    
    # Train client model (simplified)
    # ... local training code ...
    
    # Collect results
    client_params.append(client_model.state_dict())
    client_sizes.append(len(train_loader.dataset))

# Aggregate parameters
global_params = federated_averaging(client_params, client_sizes)
global_model.load_state_dict(global_params)
```

#### `add_noise_to_parameters()`

Add differential privacy noise to model parameters.

```python
def add_noise_to_parameters(parameters, noise_scale=0.005)
```

**Parameters:**
- `parameters` (OrderedDict): Model parameters
- `noise_scale` (float): Standard deviation of Gaussian noise

**Returns:**
- `OrderedDict`: Parameters with added noise

#### `get_device()`

Get the best available computing device.

```python
def get_device()
```

**Returns:**
- `torch.device`: CUDA device if available, otherwise CPU

**Example:**
```python
from training_utils import get_device

device = get_device()
print(f"Using device: {device}")

# Move model and data to device
model = model.to(device)
data = data.to(device)
```

### Classes

#### `WarmupCosineScheduler`

Custom learning rate scheduler with warmup and cosine decay.

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr=0.0)
```

**Parameters:**
- `optimizer` (torch.optim.Optimizer): Optimizer to schedule
- `num_warmup_steps` (int): Number of warmup steps
- `num_training_steps` (int): Total number of training steps
- `min_lr` (float): Minimum learning rate

**Methods:**

##### `step()`
Advance one step and update learning rates.

##### `state_dict()`
Get scheduler state for checkpointing.

##### `load_state_dict(state_dict)`
Load scheduler state from checkpoint.

**Example:**
```python
from training_utils import WarmupCosineScheduler

scheduler = WarmupCosineScheduler(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000,
    min_lr=1e-6
)

# Use in training loop
for step in range(1000):
    # Training step
    scheduler.step()  # Update learning rate
```

---

## Differential Privacy Utilities (`dp_utils.py`)

### Functions

#### `attach_privacy_engine()`

Attach Opacus PrivacyEngine to model and optimizer for DP training.

```python
def attach_privacy_engine(model, optimizer, train_loader, target_epsilon=5.0, 
                         target_delta=1e-5, noise_multiplier=None, max_grad_norm=1.0)
```

**Parameters:**
- `model` (nn.Module): PyTorch model
- `optimizer` (torch.optim.Optimizer): Optimizer to make private
- `train_loader` (DataLoader): Training data loader
- `target_epsilon` (float): Target privacy budget (ε)
- `target_delta` (float): Target privacy parameter (δ)
- `noise_multiplier` (float, optional): Noise multiplier (auto-calculated if None)
- `max_grad_norm` (float): Maximum gradient norm for clipping

**Returns:**
- `tuple`: (PrivacyEngine, private_model, private_optimizer, private_loader)

**Example:**
```python
from dp_utils import attach_privacy_engine
from models import get_model
from training_utils import get_optimizer
from dataset_loader import create_data_loaders

# Setup components
model = get_model('efficientnet')
optimizer = get_optimizer(model, 'adamw', lr=0.001)
train_loader, _, _ = create_data_loaders('MHEALTHDATASET/')

# Attach privacy engine
privacy_engine, private_model, private_optimizer, private_loader = attach_privacy_engine(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    target_epsilon=2.0,
    target_delta=1e-5,
    max_grad_norm=1.0
)

print(f"Privacy engine attached for ε={2.0}, δ={1e-5}")
```

#### `get_privacy_spent()`

Get current privacy expenditure from PrivacyEngine.

```python
def get_privacy_spent(privacy_engine, delta=1e-5)
```

**Parameters:**
- `privacy_engine` (PrivacyEngine): Opacus privacy engine
- `delta` (float): Privacy parameter for epsilon calculation

**Returns:**
- `tuple`: (epsilon, delta) privacy spent

**Example:**
```python
from dp_utils import get_privacy_spent

# During training, check privacy spent
epsilon_spent, delta = get_privacy_spent(privacy_engine, delta=1e-5)
print(f"Privacy spent: ε={epsilon_spent:.3f}, δ={delta:.2e}")

# Stop training if budget exceeded
if epsilon_spent > target_epsilon:
    print("Privacy budget exhausted, stopping training")
    break
```

#### `add_dp_noise_to_gradients()`

Manually add differential privacy noise to model gradients.

```python
def add_dp_noise_to_gradients(model, noise_scale=0.01)
```

**Parameters:**
- `model` (nn.Module): PyTorch model
- `noise_scale` (float): Standard deviation of noise

**Example:**
```python
from dp_utils import add_dp_noise_to_gradients, clip_gradients

# In training loop (manual DP implementation)
loss.backward()
clip_gradients(model, max_norm=1.0)
add_dp_noise_to_gradients(model, noise_scale=0.02)
optimizer.step()
```

#### `clip_gradients()`

Clip gradients to maximum norm for differential privacy.

```python
def clip_gradients(model, max_norm=1.0)
```

**Parameters:**
- `model` (nn.Module): PyTorch model
- `max_norm` (float): Maximum gradient norm

**Returns:**
- `float`: Total norm of gradients before clipping

#### `compute_sensitivity()`

Compute model sensitivity for differential privacy calibration.

```python
def compute_sensitivity(model, train_loader, device)
```

**Parameters:**
- `model` (nn.Module): PyTorch model
- `train_loader` (DataLoader): Training data loader
- `device` (torch.device): Computation device

**Returns:**
- `float`: Estimated model sensitivity

#### `validate_dp_parameters()`

Validate differential privacy parameters for correctness.

```python
def validate_dp_parameters(epsilon, delta)
```

**Parameters:**
- `epsilon` (float): Privacy budget
- `delta` (float): Privacy parameter

**Returns:**
- `bool`: True if parameters are valid

**Example:**
```python
from dp_utils import validate_dp_parameters

# Validate privacy parameters
if validate_dp_parameters(epsilon=2.0, delta=1e-5):
    print("Privacy parameters are valid")
else:
    print("Invalid privacy parameters")
```

#### `get_dp_recommendations()`

Get recommended DP parameters based on dataset characteristics.

```python
def get_dp_recommendations(dataset_size, batch_size, target_accuracy=0.85)
```

**Parameters:**
- `dataset_size` (int): Total number of training samples
- `batch_size` (int): Training batch size
- `target_accuracy` (float): Desired model accuracy

**Returns:**
- `dict`: Recommended privacy parameters

**Example:**
```python
from dp_utils import get_dp_recommendations

# Get DP recommendations
recommendations = get_dp_recommendations(
    dataset_size=100000,
    batch_size=512,
    target_accuracy=0.80
)

print(f"Recommended ε: {recommendations['epsilon']}")
print(f"Recommended δ: {recommendations['delta']}")
print(f"Noise multiplier: {recommendations['noise_multiplier']}")
```

#### `print_privacy_report()`

Print comprehensive privacy analysis report.

```python
def print_privacy_report(privacy_engine, round_num=0)
```

**Parameters:**
- `privacy_engine` (PrivacyEngine): Opacus privacy engine
- `round_num` (int): Current training round for reporting

**Example:**
```python
from dp_utils import print_privacy_report

# Print privacy report during training
print_privacy_report(privacy_engine, round_num=5)
```

---

## Data Partitioning (`partitioning.py`)

### Functions

#### `dirichlet_noniid_partition()`

Create non-IID data partitions using Dirichlet distribution.

```python
def dirichlet_noniid_partition(labels, num_clients, alpha, random_state=42)
```

**Parameters:**
- `labels` (np.ndarray): Array of sample labels
- `num_clients` (int): Number of federated clients
- `alpha` (float): Dirichlet concentration parameter (lower = more heterogeneous)
- `random_state` (int): Random seed for reproducibility

**Returns:**
- `List[List[int]]`: List of sample indices for each client

**Example:**
```python
from partitioning import dirichlet_noniid_partition
from dataset_loader import MHEALTHDataset

# Load dataset
dataset = MHEALTHDataset('MHEALTHDATASET/')

# Create non-IID partitions
client_indices = dirichlet_noniid_partition(
    labels=dataset.labels,
    num_clients=5,
    alpha=0.5,  # Moderate heterogeneity
    random_state=42
)

# Check partition sizes
for i, indices in enumerate(client_indices):
    print(f"Client {i+1}: {len(indices)} samples")
```

#### `iid_partition()`

Create IID (uniform random) data partitions.

```python
def iid_partition(labels, num_clients, random_state=42)
```

**Parameters:**
- `labels` (np.ndarray): Array of sample labels
- `num_clients` (int): Number of federated clients
- `random_state` (int): Random seed for reproducibility

**Returns:**
- `List[List[int]]`: List of sample indices for each client

**Example:**
```python
from partitioning import iid_partition

# Create IID partitions for comparison
iid_indices = iid_partition(
    labels=dataset.labels,
    num_clients=5,
    random_state=42
)
```

#### `analyze_partition_heterogeneity()`

Analyze data distribution heterogeneity across clients.

```python
def analyze_partition_heterogeneity(labels, client_indices)
```

**Parameters:**
- `labels` (np.ndarray): Array of sample labels
- `client_indices` (List[List[int]]): Client partition indices

**Returns:**
- `dict`: Heterogeneity analysis results

**Example:**
```python
from partitioning import analyze_partition_heterogeneity

# Analyze partition heterogeneity
analysis = analyze_partition_heterogeneity(dataset.labels, client_indices)

print(f"Mean heterogeneity: {analysis['mean_heterogeneity']:.2f}")
print(f"Std heterogeneity: {analysis['std_heterogeneity']:.2f}")

# Access detailed statistics
client_distributions = analysis['client_class_distributions']
for i, dist in enumerate(client_distributions):
    print(f"Client {i+1} class distribution: {dist}")
```

#### `print_partition_analysis()`

Print comprehensive partition analysis report.

```python
def print_partition_analysis(labels, client_indices, alpha=None)
```

**Parameters:**
- `labels` (np.ndarray): Array of sample labels
- `client_indices` (List[List[int]]): Client partition indices
- `alpha` (float, optional): Dirichlet alpha parameter for reporting

**Example:**
```python
from partitioning import print_partition_analysis

# Print detailed analysis
print_partition_analysis(
    labels=dataset.labels,
    client_indices=client_indices,
    alpha=0.5
)
```

---

## Usage Examples

### Complete Federated Learning Pipeline

```python
# Import necessary modules
from dataset_loader import create_federated_data_loaders
from models import get_model, count_parameters
from training_utils import (
    set_seed, get_optimizer, train_epoch, evaluate_model, 
    federated_averaging, get_device
)
from dp_utils import attach_privacy_engine, get_privacy_spent
import torch
import torch.nn as nn
import copy

# Setup
set_seed(42)
device = get_device()

# Create federated data loaders
client_loaders = create_federated_data_loaders(
    data_path='MHEALTHDATASET/',
    num_clients=5,
    batch_size=256,
    alpha=0.5,
    random_state=42
)

# Create global model
global_model = get_model(
    model_name='efficientnet',
    input_dim=23,
    num_classes=13,
    hidden_dim=256,
    dropout_rate=0.2
).to(device)

print(f"Model parameters: {count_parameters(global_model):,}")

# Federated training loop
criterion = nn.CrossEntropyLoss()
num_rounds = 10

for round_num in range(1, num_rounds + 1):
    print(f"\n=== Round {round_num} ===")
    
    client_params = []
    client_sizes = []
    
    # Train each client
    for client_id, (train_loader, test_loader) in enumerate(client_loaders):
        print(f"Training client {client_id + 1}")
        
        # Clone global model for client
        client_model = copy.deepcopy(global_model)
        optimizer = get_optimizer(client_model, 'adamw', lr=0.001)
        
        # Local training
        for epoch in range(3):  # 3 local epochs
            metrics = train_epoch(
                client_model, train_loader, criterion, optimizer, device
            )
        
        # Collect client results
        client_params.append(client_model.state_dict())
        client_sizes.append(len(train_loader.dataset))
        
        print(f"  Final accuracy: {metrics['accuracy']:.2f}%")
    
    # Federated aggregation
    global_params = federated_averaging(client_params, client_sizes)
    global_model.load_state_dict(global_params)
    
    # Evaluate global model
    all_correct = 0
    all_total = 0
    
    with torch.no_grad():
        for _, test_loader in client_loaders:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                all_correct += pred.eq(target.view_as(pred)).sum().item()
                all_total += target.size(0)
    
    global_accuracy = 100.0 * all_correct / all_total
    print(f"Global accuracy: {global_accuracy:.2f}%")

print("\nFederated learning completed!")
```

### Differential Privacy Integration

```python
# Extend the above example with differential privacy
from dp_utils import attach_privacy_engine, get_privacy_spent

# Privacy configuration
target_epsilon = 2.0
target_delta = 1e-5

# Modified client training with DP
for client_id, (train_loader, test_loader) in enumerate(client_loaders):
    print(f"Training client {client_id + 1} with DP")
    
    # Setup client model
    client_model = copy.deepcopy(global_model)
    optimizer = get_optimizer(client_model, 'adamw', lr=0.001)
    
    # Attach privacy engine
    privacy_engine, private_model, private_optimizer, private_loader = attach_privacy_engine(
        model=client_model,
        optimizer=optimizer,
        train_loader=train_loader,
        target_epsilon=target_epsilon / num_rounds,  # Budget per round
        target_delta=target_delta,
        max_grad_norm=1.0
    )
    
    # DP training
    for epoch in range(3):
        private_model.train()
        for batch_idx, (data, target) in enumerate(private_loader):
            data, target = data.to(device), target.to(device)
            
            private_optimizer.zero_grad()
            output = private_model(data)
            loss = criterion(output, target)
            loss.backward()
            private_optimizer.step()
    
    # Check privacy spent
    epsilon_spent, _ = get_privacy_spent(privacy_engine, target_delta)
    print(f"  Privacy spent: ε={epsilon_spent:.3f}")
    
    # Continue with federated aggregation...
```

This comprehensive API reference provides detailed documentation for all major components of the privacy-preserving federated learning framework, enabling researchers and developers to effectively utilize and extend the system.
