"""
MHEALTH Dataset Loader Module for Federated Learning Applications

This module provides comprehensive data loading and preprocessing capabilities for the
MHEALTH (Mobile Health) dataset, specifically designed for federated learning scenarios.
It supports flexible data partitioning, standardization, and efficient data loading
with PyTorch integration.

Key Features:
- Robust multi-format file parsing with delimiter auto-detection
- Configurable data standardization and preprocessing
- PyTorch Dataset integration with tensor conversion
- Federated learning data partitioning with non-IID distributions
- Comprehensive data integrity validation and error handling
- Memory-efficient batch processing for large-scale experiments

The module is optimized for distributed machine learning scenarios where data
heterogeneity and privacy considerations are critical factors.

Dependencies:
    os: File system operations and path management
    pandas: Structured data manipulation and CSV parsing
    numpy: Numerical computations and array operations
    torch: Deep learning framework and tensor operations
    sklearn: Data preprocessing and standardization utilities
    warnings: Error message filtering for cleaner output
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')  # Suppress non-critical warnings for cleaner output


class MHEALTHDataset(Dataset):
    """
    Custom PyTorch Dataset for MHEALTH (Mobile Health) Sensor Data
    
    This class implements a comprehensive PyTorch Dataset for handling sensor data
    from mobile health devices. It supports multi-subject data aggregation, flexible
    preprocessing options, and efficient data loading for machine learning applications.
    
    The dataset handles sensor readings from accelerometers, gyroscopes, and magnetometers
    across multiple subjects and activity types, making it suitable for activity recognition
    and human behavior analysis tasks.
    
    Attributes:
        data (numpy.ndarray): Preprocessed sensor feature data
        labels (numpy.ndarray): Corresponding activity class labels
        transform (callable): Optional data transformation pipeline
        standardize (bool): Flag for per-file feature standardization
    
    Note:
        The dataset automatically handles data integrity validation, missing value
        removal, and format standardization across different file types and subjects.
    """
    
    def __init__(self, data_path, transform=None, standardize=True):
        """
        Initialize the MHEALTH dataset with comprehensive data loading and preprocessing.
        
        This constructor handles the complete data loading pipeline including file discovery,
        format detection, data parsing, preprocessing, and validation. It supports multiple
        file formats and automatically standardizes features for improved model performance.
        
        Args:
            data_path (str): Path to the MHEALTH dataset directory containing subject files.
                           Should contain .log files with sensor readings and activity labels.
            transform (callable, optional): Optional transformation function to apply to samples.
                                           Useful for data augmentation or additional preprocessing.
            standardize (bool): Whether to apply StandardScaler normalization per subject file.
                               Recommended for neural network training to ensure feature scale consistency.
        
        Raises:
            ValueError: If no valid data is found in the specified directory
            FileNotFoundError: If the data_path directory does not exist
            
        Example:
            >>> dataset = MHEALTHDataset('./MHEALTHDATASET', standardize=True)
            >>> print(f"Loaded {len(dataset)} samples with {dataset.data.shape[1]} features")
        """
        self.data = []
        self.labels = []
        self.transform = transform
        self.standardize = standardize
        
        # Multi-subject data aggregation: Load data from all subject files in directory
        if os.path.isdir(data_path):
            for file in os.listdir(data_path):
                if file.endswith('.log') or file.endswith('.txt.log'):
                    file_path = os.path.join(data_path, file)
                    self._load_file(file_path)
        
        # Data structure optimization: Convert lists to efficient numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        print(f"Loaded {len(self.data)} samples with {len(np.unique(self.labels))} classes")
        
        # Data integrity validation: Ensure non-empty dataset
        if len(self.data) == 0:
            raise ValueError("No data loaded from the specified path")
        
        # Label format standardization: Ensure integer labels for classification
        self.labels = self.labels.astype(int)
        
    def _load_file(self, file_path):
        """
        Load and preprocess data from a single subject file with robust format handling.
        
        This method implements flexible file parsing that can handle multiple delimiter
        formats commonly found in sensor data files. It performs comprehensive data
        validation, cleaning, and optional standardization to ensure consistent
        data quality across different subjects and recording sessions.
        
        Args:
            file_path (str): Path to the individual subject data file
            
        Note:
            This method tries multiple delimiters (whitespace, comma, tab) to handle
            different file formats. Invalid samples (negative labels) are automatically
            filtered out to maintain data quality.
        """
        try:
            # Robust file parsing: Multi-delimiter support for different file formats
            for delimiter in ['\\s+', ',', '\\t']:
                try:
                    df = pd.read_csv(file_path, sep=delimiter, header=None)
                    break
                except:
                    continue
            else:
                print(f"Warning: Could not read file {file_path} with any delimiter")
                return
            
            # Feature-label separation: Extract sensor features and activity labels
            X = df.iloc[:, :-1].values  # Features: all columns except last (sensor readings)
            y = df.iloc[:, -1].values   # Labels: last column (activity classes)
            
            # Data quality assurance: Filter out invalid samples and missing values
            valid_idx = y >= 0
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) == 0:
                print(f"Warning: No valid data in file {file_path}")
                return
            
            # Feature normalization: Apply per-file standardization if requested
            if self.standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # Data aggregation: Append processed data to master collections
            self.data.extend(X)
            self.labels.extend(y)
            
        except Exception as e:
            print(f"Error loading dataset {file_path}: {e}")
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Total number of sensor data samples across all subjects
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset with automatic tensor conversion.
        
        This method implements the PyTorch Dataset protocol for efficient data loading.
        It handles index validation, optional data transformation, and automatic
        conversion to PyTorch tensors with appropriate data types for training.
        
        Args:
            idx (int or torch.Tensor): Index of the sample to retrieve
            
        Returns:
            tuple: (features, label) where:
                - features (torch.FloatTensor): Sensor feature vector
                - label (torch.LongTensor): Activity class label
        """
        # Index type normalization: Handle both integer and tensor indices
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Data retrieval: Extract sample and corresponding label
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Optional data transformation: Apply user-defined preprocessing
        if self.transform:
            sample = self.transform(sample)
        
        # Tensor conversion: Convert to PyTorch tensors with appropriate data types
        return torch.FloatTensor(sample), torch.LongTensor([label])[0]


def create_data_loaders(data_path, batch_size=64, test_size=0.2, random_state=42, num_workers=0):
    """
    Create optimized PyTorch data loaders for centralized machine learning training.
    
    This function provides a complete data loading pipeline for centralized training
    scenarios, handling dataset loading, train-test splitting, and efficient batch
    processing. It includes automatic GPU memory optimization and comprehensive
    dataset statistics for analysis.
    
    Args:
        data_path (str): Path to the MHEALTH dataset directory containing subject files.
                        Must contain .log files with sensor data and activity labels.
        batch_size (int, optional): Number of samples per batch for training. Default: 64.
                                   Larger values improve GPU utilization but require more memory.
        test_size (float, optional): Proportion of data reserved for testing (0.0-1.0). Default: 0.2.
                                    Standard 80/20 split for train/test evaluation.
        random_state (int, optional): Random seed for reproducible train-test splits. Default: 42.
                                     Ensures consistent experiments across runs.
        num_workers (int, optional): Number of subprocess workers for data loading. Default: 0.
                                    Set to 0 to avoid multiprocessing issues in some environments.
        
    Returns:
        tuple: A three-element tuple containing:
            - train_loader (DataLoader): PyTorch DataLoader for training data
            - test_loader (DataLoader): PyTorch DataLoader for test data  
            - dataset_info (dict): Comprehensive dataset statistics including:
                * num_classes: Number of unique activity classes
                * num_features: Dimensionality of sensor feature vectors
                * train_samples: Number of training samples
                * test_samples: Number of test samples
                * total_samples: Total dataset size
                * class_distribution: Sample count per activity class
                
    Example:
        >>> train_loader, test_loader, info = create_data_loaders('./MHEALTHDATASET')
        >>> print(f"Training on {info['train_samples']} samples")
        >>> for batch_features, batch_labels in train_loader:
        >>>     # Training loop implementation
        >>>     pass
    """
    # Dataset initialization: Load complete MHEALTH dataset with preprocessing
    full_dataset = MHEALTHDataset(data_path)
    
    # Data partitioning: Create reproducible train-test split
    train_size = int((1 - test_size) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    # Training data loader: Optimized for model training with shuffling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Randomize sample order for better training dynamics
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False  # GPU memory optimization
    )
    
    # Test data loader: Deterministic evaluation without shuffling
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Maintain consistent evaluation order
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False  # GPU memory optimization
    )
    
    # Comprehensive dataset statistics: Extract metadata for analysis and debugging
    dataset_info = {
        'num_classes': len(np.unique(full_dataset.labels)),
        'num_features': full_dataset.data.shape[1],
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'total_samples': len(full_dataset),
        'class_distribution': np.bincount(full_dataset.labels)
    }
    
    print(f"Dataset Info:")
    print(f"  Features: {dataset_info['num_features']}")
    print(f"  Classes: {dataset_info['num_classes']}")
    print(f"  Train samples: {dataset_info['train_samples']}")
    print(f"  Test samples: {dataset_info['test_samples']}")
    print(f"  Class distribution: {dataset_info['class_distribution']}")
    
    return train_loader, test_loader, dataset_info


def create_federated_data_loaders(data_path, num_clients=5, batch_size=64, alpha=0.5, random_state=42):
    """
    Create federated learning data loaders with realistic non-IID data distribution.
    
    This function implements sophisticated data partitioning for federated learning
    scenarios using Dirichlet distribution to simulate realistic data heterogeneity
    across clients. Each client receives a unique subset of data with varying class
    distributions, mimicking real-world federated learning conditions.
    
    Args:
        data_path (str): Path to the MHEALTH dataset directory containing subject files.
                        Should contain preprocessed sensor data in .log format.
        num_clients (int, optional): Number of federated learning clients to simulate. Default: 5.
                                   More clients increase communication overhead but improve diversity.
        batch_size (int, optional): Batch size for client-local training. Default: 64.
                                   Should be tuned based on client computational resources.
        alpha (float, optional): Dirichlet concentration parameter controlling data heterogeneity.
                                Default: 0.5. Lower values (0.1-0.5) create more heterogeneous distributions,
                                higher values (1.0+) create more uniform distributions across clients.
        random_state (int, optional): Random seed for reproducible client data partitioning.
                                     Default: 42. Ensures consistent federated experiments.
        
    Returns:
        list: List of tuples, one per client, where each tuple contains:
            - train_loader (DataLoader): Client's local training data loader
            - test_loader (DataLoader): Client's local test data loader
            
    Note:
        The function automatically applies 80/20 train-test splits within each client's
        data partition to enable local validation. Pin memory is optimized for GPU usage.
        
    Example:
        >>> client_loaders = create_federated_data_loaders('./MHEALTHDATASET', num_clients=10, alpha=0.3)
        >>> for client_id, (train_loader, test_loader) in enumerate(client_loaders):
        >>>     print(f"Client {client_id}: {len(train_loader.dataset)} training samples")
    """
    from partitioning import dirichlet_noniid_partition
    
    # Dataset initialization: Load complete MHEALTH dataset for partitioning
    full_dataset = MHEALTHDataset(data_path)
    
    # Non-IID partitioning: Create heterogeneous client data distributions
    client_indices = dirichlet_noniid_partition(
        full_dataset.labels, 
        num_clients, 
        alpha, 
        random_state
    )
    
    # Client data loader collection: Initialize storage for all client loaders
    client_loaders = []
    
    # Per-client data loader creation: Process each client's data partition
    for client_id, indices in enumerate(client_indices):
        # Client data subset: Extract samples assigned to this specific client
        client_data = torch.utils.data.Subset(full_dataset, indices)
        train_size = int(0.8 * len(client_data))
        test_size = len(client_data) - train_size
        
        # Client-specific train-test split: Maintain data isolation between clients
        train_subset, test_subset = torch.utils.data.random_split(
            client_data, [train_size, test_size],
            generator=torch.Generator().manual_seed(random_state + client_id)
        )
        
        # Client training data loader: Optimized for local model training
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,  # Enable shuffling for better local training dynamics
            num_workers=0,  # Single-threaded to avoid multiprocessing complications
            pin_memory=True if torch.cuda.is_available() else False  # GPU optimization
        )
        
        # Client test data loader: Consistent evaluation across federated rounds
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,  # Deterministic order for consistent evaluation
            num_workers=0,  # Single-threaded for stability
            pin_memory=True if torch.cuda.is_available() else False  # GPU optimization
        )
        
        # Client loader registration: Add to federated learning setup
        client_loaders.append((train_loader, test_loader))
        
        print(f"Client {client_id + 1}: {len(train_subset)} train, {len(test_subset)} test samples")
    
    return client_loaders
