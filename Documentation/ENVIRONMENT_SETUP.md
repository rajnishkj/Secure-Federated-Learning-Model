# Environment Setup and Installation Guide

## Overview

This guide provides comprehensive instructions for setting up the development environment for the Privacy-Preserving Machine Learning for IoT Data project. The setup supports both local development and cloud-based execution (Google Colab).

## System Requirements

### Hardware Requirements

#### Minimum Specifications
```
CPU: 4 cores, 2.0 GHz
RAM: 8 GB
Storage: 10 GB available space
Network: Broadband internet connection
GPU: Optional but recommended
```

#### Recommended Specifications
```
CPU: 8+ cores, 3.0+ GHz (Intel i7/AMD Ryzen 7 or equivalent)
RAM: 16 GB or higher
Storage: 20 GB SSD available space
GPU: NVIDIA GPU with 8+ GB VRAM (RTX 3070 or equivalent)
Network: High-speed internet for dataset downloads
```

#### Optimal Specifications (for large-scale experiments)
```
CPU: 16+ cores, 3.5+ GHz (Intel i9/AMD Ryzen 9 or equivalent)
RAM: 32 GB or higher
Storage: 50 GB NVMe SSD
GPU: NVIDIA RTX 4080/4090 or A100 with 16+ GB VRAM
Network: Gigabit ethernet or high-speed WiFi
```

### Software Requirements

#### Operating System Support
- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 10.15+ (Intel or Apple Silicon)
- **Linux**: Ubuntu 18.04+ / CentOS 7+ / Debian 10+

#### Python Environment
- **Python Version**: 3.8, 3.9, or 3.10 (recommended: 3.9)
- **Package Manager**: pip (latest version) or conda

## Installation Methods

### Method 1: Local Environment Setup (Recommended for Development)

#### Step 1: Python Environment Setup

**Option A: Using Conda (Recommended)**
```bash
# Install Miniconda (if not already installed)
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create new environment
conda create -n privacy-ml python=3.9
conda activate privacy-ml

# Verify installation
python --version  # Should show Python 3.9.x
```

**Option B: Using Python venv**
```bash
# Create virtual environment
python -m venv privacy-ml-env

# Activate environment
# On Windows:
privacy-ml-env\Scripts\activate
# On macOS/Linux:
source privacy-ml-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 2: Clone Repository
```bash
# Clone the repository
git clone <repository-url>
cd Privacy_ML_Project

# Verify project structure
ls -la  # Should show all project files
```

#### Step 3: Install Dependencies

**Core Dependencies Installation**
```bash
# Install PyTorch (adjust CUDA version as needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (if no GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Opacus for differential privacy
pip install opacus>=1.5.0

# Install remaining dependencies
pip install -r requirements.txt
```

**Alternative: All-in-one Installation**
```bash
# Install everything with versions specified in requirements.txt
pip install -r requirements.txt

# Verify critical packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import opacus; print(f'Opacus: {opacus.__version__}')"
```

#### Step 4: Verify Installation
```bash
# Run verification script
python -c "
import torch
import opacus
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn

print('✅ All core dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Opacus: {opacus.__version__}')
"
```

### Method 2: Google Colab Setup (Recommended for Experimentation)

#### Quick Setup for Colab

**Step 1: Prepare Google Drive**
```python
# Create folder structure in Google Drive:
# MyDrive/
# └── Privacy_ML_Project/
#     ├── dataset_loader.py
#     ├── models.py
#     ├── training_utils.py
#     ├── dp_utils.py
#     ├── partitioning.py
#     ├── MHEALTHDATASET/
#     │   ├── mHealth_subject1.log
#     │   └── ... (all 10 subject files)
#     └── requirements.txt
```

**Step 2: Colab Notebook Setup**
```python
# Copy this cell into each notebook as Cell 1
# ============================================================================
# GOOGLE COLAB SETUP CELL - COPY THIS INTO EACH NOTEBOOK AS CELL 1
# ============================================================================

from google.colab import drive
import os
import torch
import time

print("Setting up Google Colab environment...")

# 1. Mount Google Drive
drive.mount('/content/drive')
print("Google Drive mounted successfully!")

# 2. Navigate to project directory
project_path = '/content/drive/MyDrive/Privacy_ML_Project'
try:
    os.chdir(project_path)
    print(f"Navigated to: {os.getcwd()}")
except FileNotFoundError:
    print(f"Project folder not found: {project_path}")
    print("Please create 'Privacy_ML_Project' folder in your Google Drive and upload all files!")
    raise

# 3. Install required packages
print("Installing required packages...")
!pip install -q opacus mlflow seaborn scikit-learn tqdm typing-extensions

# 4. Verify GPU availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f"GPU DETECTED: {gpu_name} ({gpu_memory}GB)")
else:
    print("No GPU detected - using CPU")

print("Setup complete! You can now run your notebook cells.")
```

### Method 3: Docker Setup (Advanced Users)

#### Dockerfile Configuration
```dockerfile
# Create Dockerfile in project root
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

#### Docker Commands
```bash
# Build Docker image
docker build -t privacy-ml .

# Run with GPU support
docker run --gpus all -p 8888:8888 -v $(pwd):/app privacy-ml

# Run CPU-only
docker run -p 8888:8888 -v $(pwd):/app privacy-ml
```

## Dataset Setup

### MHEALTH Dataset Download and Preparation

#### Automatic Download (Recommended)
```python
# Create download script: download_dataset.py
import os
import urllib.request
import zipfile

def download_mhealth_dataset():
    """Download and extract MHEALTH dataset."""
    
    # Dataset URL (UCI Machine Learning Repository)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
    
    print("Downloading MHEALTH dataset...")
    urllib.request.urlretrieve(url, "MHEALTHDATASET.zip")
    
    print("Extracting dataset...")
    with zipfile.ZipFile("MHEALTHDATASET.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Clean up
    os.remove("MHEALTHDATASET.zip")
    
    print("Dataset downloaded and extracted successfully!")
    print(f"Dataset location: {os.path.abspath('MHEALTHDATASET')}")
    
    # Verify dataset
    dataset_files = os.listdir("MHEALTHDATASET")
    print(f"Found {len(dataset_files)} files:")
    for file in sorted(dataset_files):
        print(f"  - {file}")

if __name__ == "__main__":
    download_mhealth_dataset()
```

```bash
# Run download script
python download_dataset.py
```

#### Manual Download
```bash
# Alternative: Manual download
# 1. Visit: https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset
# 2. Download MHEALTHDATASET.zip
# 3. Extract to project root directory
# 4. Verify folder structure:

# Expected structure:
# MHEALTHDATASET/
# ├── mHealth_subject1.log
# ├── mHealth_subject2.log
# ├── ...
# ├── mHealth_subject10.log
# └── README.txt
```

#### Dataset Verification
```python
# Create verification script: verify_dataset.py
import os
import pandas as pd

def verify_mhealth_dataset():
    """Verify MHEALTH dataset integrity."""
    
    dataset_path = "MHEALTHDATASET"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset directory not found!")
        return False
    
    # Check for all subject files
    expected_files = [f"mHealth_subject{i}.log" for i in range(1, 11)]
    missing_files = []
    
    for file in expected_files:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            # Check file size and basic structure
            try:
                df = pd.read_csv(file_path, sep='\s+', header=None)
                if df.shape[1] != 24:  # 23 features + 1 label
                    print(f"⚠️  {file}: Unexpected number of columns ({df.shape[1]})")
                else:
                    print(f"✅ {file}: {df.shape[0]} samples, {df.shape[1]} columns")
            except Exception as e:
                print(f"❌ {file}: Error reading file - {e}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Dataset verification complete!")
    return True

if __name__ == "__main__":
    verify_mhealth_dataset()
```

```bash
# Verify dataset
python verify_dataset.py
```

## Development Environment Configuration

### IDE Setup

#### Visual Studio Code (Recommended)
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./privacy-ml-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.git": true,
        "**/node_modules": true,
        "**/.DS_Store": true
    }
}
```

**Recommended Extensions**:
- Python (Microsoft)
- Pylance (Microsoft)
- Jupyter (Microsoft)
- GitLens (GitKraken)
- Black Formatter (Microsoft)

#### PyCharm Configuration
```python
# Configure PyCharm:
# 1. File → Settings → Project → Python Interpreter
# 2. Add interpreter → Existing environment
# 3. Select: privacy-ml-env/bin/python
# 4. Apply and restart
```

#### Jupyter Notebook Setup
```bash
# Install Jupyter extensions
pip install jupyter jupyter-contrib-nbextensions

# Enable extensions
jupyter contrib nbextension install --user
jupyter nbextension enable --py widgetsnbextension

# Start Jupyter
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

### Environment Variables
```bash
# Create .env file in project root
cat > .env << EOF
# Project Configuration
PROJECT_NAME=Privacy_ML_Project
PYTHONPATH=.

# Hardware Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_HOME=./torch_cache

# Experiment Configuration
RANDOM_SEED=42
DEFAULT_DEVICE=auto

# Privacy Configuration
DEFAULT_EPSILON=2.0
DEFAULT_DELTA=1e-5

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=experiment.log
EOF
```

```python
# Load environment variables in Python
from dotenv import load_dotenv
import os

load_dotenv()

# Access variables
PROJECT_NAME = os.getenv('PROJECT_NAME', 'Privacy_ML_Project')
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
```

## Troubleshooting

### Common Installation Issues

#### Issue 1: PyTorch CUDA Compatibility
```bash
# Problem: CUDA version mismatch
# Solution: Check CUDA version and install compatible PyTorch
nvcc --version  # Check CUDA version

# Install appropriate PyTorch version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Issue 2: Opacus Installation Problems
```bash
# Problem: Opacus compatibility issues
# Solution: Use specific version
pip uninstall opacus
pip install opacus==1.5.0

# Alternative: Install from source
pip install git+https://github.com/pytorch/opacus.git
```

#### Issue 3: Memory Issues
```python
# Problem: Out of memory during training
# Solution: Reduce batch size and enable memory optimization

# In your training configuration:
BATCH_SIZE = 128  # Reduce from 512
MAX_PHYSICAL_BATCH_SIZE = 32  # For Opacus BatchMemoryManager

# Enable memory-efficient options
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()  # Clear cache between experiments
```

#### Issue 4: Dataset Loading Errors
```python
# Problem: Dataset files not found or corrupted
# Solution: Verify file paths and formats

import os
import pandas as pd

def diagnose_dataset_issues():
    """Diagnose common dataset problems."""
    
    dataset_path = "MHEALTHDATASET"
    
    # Check directory existence
    if not os.path.exists(dataset_path):
        print(f"❌ Directory not found: {dataset_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return
    
    # Check individual files
    for i in range(1, 11):
        file_path = os.path.join(dataset_path, f"mHealth_subject{i}.log")
        if os.path.exists(file_path):
            try:
                # Try different delimiters
                for sep in ['\s+', ',', '\t']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, header=None, nrows=5)
                        print(f"✅ {file_path}: Successfully read with separator '{sep}'")
                        print(f"   Shape: {df.shape}")
                        break
                    except:
                        continue
                else:
                    print(f"❌ {file_path}: Could not read with any separator")
            except Exception as e:
                print(f"❌ {file_path}: Error - {e}")
        else:
            print(f"❌ {file_path}: File not found")

# Run diagnostics
diagnose_dataset_issues()
```

### Performance Optimization

#### GPU Optimization
```python
# Optimize GPU usage
import torch

def optimize_gpu_settings():
    """Configure optimal GPU settings."""
    
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Memory management
        torch.cuda.empty_cache()
        
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        return True
    else:
        print("No GPU available - using CPU")
        return False

# Apply optimizations
gpu_available = optimize_gpu_settings()
```

#### Memory Management
```python
# Memory-efficient training configuration
MEMORY_EFFICIENT_CONFIG = {
    'batch_size': 256,  # Smaller batches
    'max_physical_batch_size': 32,  # For Opacus
    'gradient_accumulation_steps': 2,  # Simulate larger batches
    'pin_memory': True,  # Faster GPU transfer
    'num_workers': 0,  # Avoid multiprocessing issues
}
```

## Testing and Validation

### Unit Tests
```bash
# Create test directory structure
mkdir tests
touch tests/__init__.py

# Run basic tests
python -m pytest tests/ -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
```

### Integration Tests
```python
# Create integration test: test_integration.py
import torch
from dataset_loader import create_data_loaders
from models import get_model
from training_utils import train_epoch, get_optimizer

def test_end_to_end_training():
    """Test complete training pipeline."""
    
    # Load data
    train_loader, test_loader, info = create_data_loaders(
        data_path="MHEALTHDATASET",
        batch_size=32,
        test_size=0.2
    )
    
    # Create model
    model = get_model(
        model_name='efficientnet',
        input_dim=info['num_features'],
        num_classes=info['num_classes'],
        hidden_dim=64,  # Smaller for testing
        dropout_rate=0.1
    )
    
    # Train for one epoch
    optimizer = get_optimizer(model, 'adamw', lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')  # Use CPU for testing
    
    metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Verify training worked
    assert metrics['accuracy'] > 0
    assert metrics['loss'] < 10  # Reasonable loss value
    
    print("✅ Integration test passed!")

if __name__ == "__main__":
    test_end_to_end_training()
```

### Performance Benchmarking
```python
# Create benchmark script: benchmark.py
import time
import torch
from dataset_loader import create_data_loaders
from models import get_model

def benchmark_training_speed():
    """Benchmark training performance."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, _, info = create_data_loaders(
        data_path="MHEALTHDATASET",
        batch_size=512,
        test_size=0.8  # Use only 20% for benchmarking
    )
    
    # Create model
    model = get_model(
        model_name='efficientnet',
        input_dim=info['num_features'],
        num_classes=info['num_classes']
    ).to(device)
    
    # Benchmark training speed
    model.train()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    sample_count = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 10:  # Benchmark 10 batches
            break
            
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        sample_count += data.size(0)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    samples_per_second = sample_count / elapsed_time
    
    print(f"Device: {device}")
    print(f"Samples processed: {sample_count}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Throughput: {samples_per_second:.1f} samples/second")

if __name__ == "__main__":
    benchmark_training_speed()
```

## Production Deployment Considerations

### Security Best Practices
```bash
# Environment isolation
python -m venv production-env
source production-env/bin/activate

# Install exact versions
pip install -r requirements-production.txt

# Secure configuration
export PYTHONHASHSEED=random
export PYTHONDONTWRITEBYTECODE=1
```

### Monitoring and Logging
```python
# Configure logging
import logging
import sys

def setup_logging():
    """Configure production logging."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('privacy_ml.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure specific loggers
    logging.getLogger('opacus').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Usage
logger = setup_logging()
logger.info("Privacy-preserving ML system initialized")
```

### Deployment Checklist
```bash
# Pre-deployment verification
✅ All dependencies installed and tested
✅ Dataset integrity verified
✅ Model training pipeline tested
✅ Privacy mechanisms validated
✅ Performance benchmarks completed
✅ Security configurations applied
✅ Logging and monitoring configured
✅ Backup and recovery procedures tested
✅ Documentation updated and complete
✅ Code review completed
```

This comprehensive setup guide ensures a robust development environment for privacy-preserving machine learning research and deployment.
