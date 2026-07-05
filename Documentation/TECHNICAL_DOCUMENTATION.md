# Technical Documentation

## System Architecture

### Overview

The privacy-preserving federated learning system implements a multi-layered architecture designed for secure, decentralized machine learning on IoT sensor data. The system integrates differential privacy mechanisms with federated learning protocols to provide formal privacy guarantees while maintaining model utility.

### Core Components

#### 1. Data Layer (`dataset_loader.py`)

**Purpose**: Handles MHEALTH dataset loading, preprocessing, and federated partitioning.

**Key Features**:
- Multi-format file support (whitespace, comma, tab delimited)
- Automatic standardization using sklearn StandardScaler
- Non-IID partitioning using Dirichlet distribution
- Memory-efficient data loading with PyTorch DataLoader

**Data Pipeline**:
```
Raw Sensor Files → Data Cleaning → Feature Standardization → Federated Partitioning → DataLoaders
```

**Technical Specifications**:
- Input: 23-dimensional sensor vectors (accelerometer, gyroscope, magnetometer)
- Output: 13-class activity classification
- Preprocessing: Z-score normalization per subject
- Partitioning: Dirichlet(α) distribution for non-IID splits

#### 2. Model Layer (`models.py`)

**Purpose**: Neural network architectures optimized for time-series sensor data.

**Available Architectures**:

##### EfficientNet (Primary Model)
```python
Architecture: 23 → 256 → 256 → 128 → 64 → 13
Components:
- Fully connected layers with batch normalization
- Residual connections for gradient flow
- Dropout regularization (rate: 0.2-0.3)
- Xavier weight initialization
- ReLU activation functions
```

##### Conv1DNet (Alternative)
```python
Architecture: 1D Convolutions + Global Average Pooling
Layers:
- Conv1D: 1→64→128→64→32 channels
- Kernel sizes: 3, 3, 3, 1
- Batch normalization after each convolution
- AdaptiveAvgPool1d for global feature aggregation
```

##### TransformerNet (Experimental)
```python
Architecture: Transformer encoder with positional encoding
Components:
- Input projection: 23 → 128 dimensions
- Multi-head attention: 8 heads
- 2 transformer encoder layers
- Feedforward dimension: 512
```

**Model Selection Criteria**:
- Parameter efficiency for federated learning
- Convergence stability under non-IID data
- Compatibility with differential privacy mechanisms

#### 3. Training Layer (`training_utils.py`)

**Purpose**: Training utilities, optimizers, and federated learning protocols.

**Core Functions**:

##### Federated Averaging (FedAvg)
```python
def federated_averaging(client_parameters, client_sizes):
    # Weighted average based on client data sizes
    total_size = sum(client_sizes)
    for param in parameters:
        global_param = sum(weight * client_param for weight, client_param)
    return global_parameters
```

##### Training Loop
```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    # Standard supervised training with gradient clipping
    # Supports learning rate scheduling
    # Returns training metrics (loss, accuracy)
```

##### Optimization Configuration
- **Primary Optimizer**: AdamW with decoupled weight decay
- **Learning Rate**: 0.001-0.004 (adaptive based on privacy budget)
- **Weight Decay**: 0.001-0.01 (regularization)
- **Gradient Clipping**: L2-norm clipping at 1.0
- **Batch Size**: 256-512 (memory optimized)

#### 4. Privacy Layer (`dp_utils.py`)

**Purpose**: Differential privacy implementation using Opacus framework.

**Privacy Mechanisms**:

##### Differential Privacy Integration
```python
def attach_privacy_engine(model, optimizer, train_loader, 
                         target_epsilon, target_delta, max_grad_norm):
    # Opacus privacy engine attachment
    # Gradient clipping and noise addition
    # Privacy accounting using RDP
```

##### Privacy Budget Management
```python
Privacy Parameters:
- Epsilon (ε): 1.0, 2.0, 5.0 (privacy budgets tested)
- Delta (δ): 1e-5 (failure probability)
- Clipping Norm: 1.0-1.1 (gradient bound)
- Noise Multiplier: Auto-calculated based on ε and δ
```

##### Privacy Accounting
- **Mechanism**: Rényi Differential Privacy (RDP)
- **Composition**: Advanced composition for multiple queries
- **Tracking**: Real-time privacy budget consumption
- **Verification**: Formal privacy guarantee validation

#### 5. Partitioning Layer (`partitioning.py`)

**Purpose**: Non-IID data distribution simulation for federated learning.

**Partitioning Strategies**:

##### Dirichlet Non-IID Partitioning
```python
def dirichlet_noniid_partition(labels, num_clients, alpha):
    # Generate Dirichlet distribution for each class
    # Alpha controls heterogeneity (lower = more heterogeneous)
    # Allocate samples based on class probabilities
```

**Heterogeneity Analysis**:
- **Metrics**: Class distribution entropy, KL-divergence
- **Visualization**: Class distribution heatmaps
- **Validation**: Statistical tests for non-IID verification

### Communication Protocol

#### Federated Learning Protocol
```
1. Server initializes global model
2. Broadcast global model to selected clients
3. Clients perform local training with DP
4. Clients send model updates (gradients/parameters)
5. Server aggregates updates using FedAvg
6. Repeat for multiple rounds
```

#### Privacy-Preserving Communication
```
1. Client-side gradient clipping
2. Gaussian noise addition (calibrated to ε, δ)
3. Secure aggregation (weighted averaging)
4. Privacy budget tracking per client
5. Early stopping on budget exhaustion
```

### Performance Optimizations

#### Memory Management
- **Batch Memory Manager**: Opacus integration for memory-efficient DP
- **Gradient Accumulation**: Support for large effective batch sizes
- **Model Checkpointing**: Automatic state saving and recovery

#### Computational Efficiency
- **GPU Acceleration**: CUDA support with automatic device detection
- **Mixed Precision**: Optional FP16 training for speed improvements
- **Parallel Processing**: Multi-core data loading with pin_memory

#### Communication Efficiency
- **Parameter Compression**: Gradient quantization options
- **Selective Updates**: Send only significant parameter changes
- **Asynchronous Updates**: Non-blocking client communications

### Security Considerations

#### Privacy Guarantees
- **Formal Definition**: (ε, δ)-differential privacy
- **Threat Model**: Honest-but-curious server and clients
- **Attack Resistance**: Protection against inference attacks

#### Implementation Security
- **Secure Random Generation**: Cryptographically secure noise
- **Memory Safety**: No parameter leakage in memory
- **Audit Trail**: Complete privacy accounting logs

## Experimental Framework

### Evaluation Metrics

#### Model Performance
- **Accuracy**: Classification accuracy on test sets
- **Loss**: Cross-entropy loss convergence
- **F1-Score**: Balanced performance across classes
- **Confusion Matrix**: Detailed classification analysis

#### Privacy Metrics
- **Privacy Budget**: Total ε consumed per client
- **Noise Scale**: Actual noise added to gradients
- **Privacy Loss**: Utility degradation due to privacy

#### System Metrics
- **Communication Rounds**: Convergence speed
- **Training Time**: Computational efficiency
- **Memory Usage**: Resource consumption

### Experimental Setup

#### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 16GB VRAM
- **Optimal**: Multi-GPU setup for large-scale experiments

#### Software Dependencies
```python
Core Dependencies:
- PyTorch 2.0+
- Opacus 1.5.0
- NumPy 1.20+
- Pandas 2.0+
- Scikit-learn 1.0+

Visualization:
- Matplotlib 3.5+
- Seaborn 0.11+

Development:
- Jupyter Notebook
- MLflow (experiment tracking)
```

## Implementation Details

### Model Training Process

#### Centralized Training (Baseline)
```python
1. Load complete MHEALTH dataset
2. Split into train/validation/test
3. Train EfficientNet model
4. Evaluate performance metrics
5. Save baseline results
```

#### Federated Learning Training
```python
1. Partition data using Dirichlet distribution
2. Initialize global model on server
3. For each federated round:
   a. Sample client subset
   b. Broadcast global model
   c. Local training on each client
   d. Aggregate model updates
   e. Update global model
4. Evaluate final model performance
```

#### Differential Privacy Training
```python
1. Set up privacy budget (ε, δ)
2. Configure Opacus privacy engine
3. For each client training:
   a. Clip gradients to max norm
   b. Add calibrated Gaussian noise
   c. Track privacy expenditure
4. Aggregate noisy updates
5. Monitor privacy budget consumption
```

### Error Handling and Robustness

#### Data Loading Errors
- **File Format Issues**: Multiple delimiter support
- **Missing Data**: Automatic filtering and imputation
- **Corrupted Files**: Graceful error handling and logging

#### Training Stability
- **Gradient Explosion**: Automatic gradient clipping
- **Loss Divergence**: Learning rate adaptation
- **Memory Issues**: Batch size reduction and cleanup

#### Privacy Mechanism Failures
- **Opacus Errors**: Fallback to manual DP implementation
- **Budget Exhaustion**: Automatic training termination
- **Noise Overflow**: Dynamic noise scaling adjustment

### Configuration Management

#### Hyperparameter Configuration
```python
# Model Configuration
MODEL_CONFIG = {
    'architecture': 'efficientnet',
    'hidden_dim': 256,
    'dropout_rate': 0.2,
    'num_classes': 13,
    'input_dim': 23
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 512,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'num_epochs': 10,
    'gradient_clip': 1.0
}

# Federated Learning Configuration
FL_CONFIG = {
    'num_clients': 5,
    'num_rounds': 16,
    'local_epochs': 3,
    'client_fraction': 1.0,
    'dirichlet_alpha': 0.5
}

# Privacy Configuration
DP_CONFIG = {
    'epsilon': [1.0, 2.0, 5.0],
    'delta': 1e-5,
    'max_grad_norm': 1.0,
    'noise_multiplier': 'auto'
}
```

#### Environment Configuration
- **Device Selection**: Automatic GPU/CPU detection
- **Random Seeds**: Reproducible experimental setup
- **Logging**: Configurable verbosity levels
- **Checkpointing**: Automatic model state saving

## Future Technical Enhancements

### Scalability Improvements
- **Hierarchical Federated Learning**: Multi-level aggregation
- **Asynchronous Protocols**: Non-blocking client updates
- **Adaptive Sampling**: Dynamic client selection strategies

### Privacy Enhancements
- **Local Differential Privacy**: Client-side privacy guarantees
- **Secure Aggregation**: Cryptographic protocol integration
- **Privacy Amplification**: Subsampling and shuffling mechanisms

### Performance Optimizations
- **Model Compression**: Pruning and quantization techniques
- **Communication Compression**: Gradient compression algorithms
- **Adaptive Learning**: Personalized federated learning approaches
