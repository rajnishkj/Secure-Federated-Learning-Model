"""
Advanced Data Partitioning Module for Federated Learning Applications

This module provides sophisticated data partitioning algorithms specifically designed
for federated learning scenarios. It implements both IID and non-IID data distribution
strategies to simulate realistic federated learning environments where client data
exhibits varying degrees of statistical heterogeneity.

Key Features:
- Dirichlet distribution-based non-IID partitioning for realistic heterogeneity simulation
- IID partitioning for baseline comparisons and controlled experiments
- Comprehensive heterogeneity analysis and visualization tools
- Configurable concentration parameters for fine-tuning data distribution
- Statistical validation and analysis of partition quality
- Support for multi-class classification scenarios

The module is essential for federated learning research as it enables the simulation
of real-world conditions where different clients possess data with varying statistical
properties, which is crucial for evaluating federated algorithm robustness and
developing privacy-preserving machine learning solutions.

Mathematical Foundation:
- Dirichlet distribution provides natural mechanism for generating heterogeneous partitions
- Concentration parameter (alpha) controls the degree of heterogeneity
- Lower alpha values create more skewed distributions across clients
- Higher alpha values approach uniform (IID) distribution

Dependencies:
    numpy: Numerical computations and random number generation
    typing: Type hints for robust code documentation
    random: Additional randomization utilities for data shuffling
"""

import numpy as np
from typing import List, Tuple
import random


def dirichlet_noniid_partition(labels: np.ndarray, num_clients: int, alpha: float, random_state: int = 42) -> List[List[int]]:
    """
    Create realistic non-IID data partitions using Dirichlet distribution.
    
    This function implements sophisticated data partitioning that simulates real-world
    federated learning scenarios where clients possess data with heterogeneous class
    distributions. It uses the Dirichlet distribution to generate concentration
    parameters that control the statistical heterogeneity across federated clients.
    
    The algorithm operates by:
    1. Generating client-specific class preferences using Dirichlet sampling
    2. Distributing samples of each class according to these preferences
    3. Ensuring each client receives a coherent subset reflecting their preferences
    4. Maintaining reproducibility through controlled random seeding
    
    Args:
        labels (np.ndarray): Complete array of sample labels for partitioning.
                           Should contain integer class labels from 0 to num_classes-1.
        num_clients (int): Number of federated clients to create partitions for.
                          More clients generally reduce average data per client.
        alpha (float): Dirichlet concentration parameter controlling heterogeneity.
                      Lower values (0.1-0.5) create highly heterogeneous distributions.
                      Higher values (1.0+) approach uniform (IID) distribution.
                      Typical values: 0.1 (very heterogeneous), 0.5 (moderate), 1.0 (mild).
        random_state (int, optional): Random seed for reproducible partitioning. Default: 42.
                                    Ensures consistent experiments across multiple runs.
        
    Returns:
        List[List[int]]: List of sample indices for each client, where:
                        - Outer list has length num_clients
                        - Inner lists contain indices of samples assigned to each client
                        - Indices refer to positions in the original dataset
                        
    Example:
        >>> labels = np.array([0, 0, 1, 1, 1, 2, 2])
        >>> partitions = dirichlet_noniid_partition(labels, num_clients=3, alpha=0.5)
        >>> print(f"Client 0 gets samples: {partitions[0]}")
        >>> print(f"Client 1 gets samples: {partitions[1]}")
        >>> print(f"Client 2 gets samples: {partitions[2]}")
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Create label distribution for each client using Dirichlet distribution
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Assign samples to clients based on label distribution
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx in range(num_classes):
        # Get indices of samples with this class
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)
        
        # Distribute samples according to Dirichlet distribution
        proportions = label_distribution[class_idx]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        proportions = np.append(proportions, len(class_indices))
        
        # Assign samples to clients
        for client_idx in range(num_clients):
            start_idx = proportions[client_idx - 1] if client_idx > 0 else 0
            end_idx = proportions[client_idx]
            client_indices[client_idx].extend(class_indices[start_idx:end_idx])
    
    # Shuffle indices within each client
    for client_idx in range(num_clients):
        random.shuffle(client_indices[client_idx])
    
    return client_indices


def iid_partition(labels: np.ndarray, num_clients: int, random_state: int = 42) -> List[List[int]]:
    """
    Partition data using IID distribution (random split).
    
    Args:
        labels (np.ndarray): Array of labels for all samples
        num_clients (int): Number of clients to partition data among
        random_state (int): Random seed for reproducibility
        
    Returns:
        List[List[int]]: List of indices for each client
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    num_samples = len(labels)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    # Split indices evenly among clients
    samples_per_client = num_samples // num_clients
    client_indices = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else num_samples
        client_indices.append(indices[start_idx:end_idx])
    
    return client_indices


def analyze_partition_heterogeneity(labels: np.ndarray, client_indices: List[List[int]]) -> dict:
    """
    Analyze the heterogeneity of data partitioning.
    
    Args:
        labels (np.ndarray): Array of labels for all samples
        client_indices (List[List[int]]): List of indices for each client
        
    Returns:
        dict: Analysis results including class distribution per client
    """
    num_classes = len(np.unique(labels))
    num_clients = len(client_indices)
    
    # Calculate class distribution for each client
    client_class_distributions = []
    for client_idx in range(num_clients):
        client_labels = labels[client_indices[client_idx]]
        class_counts = np.bincount(client_labels, minlength=num_classes)
        client_class_distributions.append(class_counts)
    
    # Calculate heterogeneity metrics
    total_samples_per_class = np.bincount(labels, minlength=num_classes)
    ideal_samples_per_client_per_class = total_samples_per_class / num_clients
    
    heterogeneity_scores = []
    for client_dist in client_class_distributions:
        # Calculate L1 distance from ideal uniform distribution
        distance = np.sum(np.abs(client_dist - ideal_samples_per_client_per_class))
        heterogeneity_scores.append(distance)
    
    analysis = {
        'client_class_distributions': client_class_distributions,
        'total_samples_per_class': total_samples_per_class,
        'ideal_samples_per_client_per_class': ideal_samples_per_client_per_class,
        'heterogeneity_scores': heterogeneity_scores,
        'mean_heterogeneity': np.mean(heterogeneity_scores),
        'std_heterogeneity': np.std(heterogeneity_scores)
    }
    
    return analysis


def print_partition_analysis(labels: np.ndarray, client_indices: List[List[int]], alpha: float = None):
    """
    Print detailed analysis of data partitioning.
    
    Args:
        labels (np.ndarray): Array of labels for all samples
        client_indices (List[List[int]]): List of indices for each client
        alpha (float): Dirichlet alpha parameter (if applicable)
    """
    analysis = analyze_partition_heterogeneity(labels, client_indices)
    
    print(f"\n=== Data Partitioning Analysis ===")
    if alpha is not None:
        print(f"Dirichlet Alpha: {alpha}")
    print(f"Number of clients: {len(client_indices)}")
    print(f"Number of classes: {len(analysis['total_samples_per_class'])}")
    print(f"Total samples: {len(labels)}")
    print(f"Mean heterogeneity score: {analysis['mean_heterogeneity']:.2f}")
    print(f"Std heterogeneity score: {analysis['std_heterogeneity']:.2f}")
    
    print(f"\nClass distribution per client:")
    for client_idx, dist in enumerate(analysis['client_class_distributions']):
        print(f"  Client {client_idx + 1}: {dist}")
    
    print(f"\nIdeal uniform distribution: {analysis['ideal_samples_per_client_per_class']}")
    print(f"Total samples per class: {analysis['total_samples_per_class']}")
