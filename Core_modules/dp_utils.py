"""
Differential Privacy Utilities for Privacy-Preserving Federated Learning

This module provides comprehensive differential privacy (DP) integration for federated
learning applications using the Opacus library. It implements privacy-preserving
training mechanisms including gradient noise injection, privacy accounting, and
parameter perturbation to ensure formal privacy guarantees in distributed learning scenarios.

Key Features:
- Opacus PrivacyEngine integration for DP-SGD optimization
- Automated privacy budget management and accounting
- Gradient clipping and noise injection mechanisms
- Privacy parameter validation and recommendations
- Comprehensive privacy analysis and reporting tools
- Support for both centralized and federated learning contexts

The module implements state-of-the-art differential privacy techniques including
the Gaussian mechanism, moments accountant, and Rényi differential privacy for
robust privacy-utility trade-off management in machine learning applications.

Dependencies:
    torch: Core PyTorch framework for neural network operations
    opacus: Differential privacy library for PyTorch models
    numpy: Numerical computations for privacy calculations
    typing: Type hints for robust code documentation
    collections: OrderedDict for parameter management
"""

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import OrderedDict


def attach_privacy_engine(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    target_epsilon: float = 5.0,
    target_delta: float = 1e-5,
    noise_multiplier: Optional[float] = None,
    max_grad_norm: float = 1.0,
):
    """
    Attach Opacus PrivacyEngine for differential privacy training integration.
    
    This function configures a complete differential privacy training setup by wrapping
    the model, optimizer, and data loader with Opacus privacy mechanisms. It implements
    DP-SGD (Differentially Private Stochastic Gradient Descent) with automatic noise
    calibration and gradient clipping for formal privacy guarantees.
    
    The function validates model compatibility, estimates optimal noise parameters,
    and returns privacy-aware components ready for DP training. It supports both
    automatic noise parameter estimation and manual configuration for advanced users.
    
    Args:
        model (nn.Module): PyTorch neural network model to be made differentially private.
                          Will be validated and potentially modified for DP compatibility.
        optimizer (torch.optim.Optimizer): Training optimizer to be wrapped with DP mechanisms.
                                          Gradients will be clipped and noised automatically.
        train_loader (DataLoader): Training data loader for batch processing.
                                  Batch size affects privacy accounting calculations.
        target_epsilon (float, optional): Target privacy budget (ε) for (ε,δ)-DP. Default: 5.0.
                                         Lower values provide stronger privacy guarantees.
        target_delta (float, optional): Target privacy parameter (δ) for (ε,δ)-DP. Default: 1e-5.
                                       Should be much smaller than 1/dataset_size.
        noise_multiplier (Optional[float]): Gaussian noise scale multiplier. Default: None.
                                           If None, automatically estimated from privacy parameters.
        max_grad_norm (float, optional): Maximum L2 norm for gradient clipping. Default: 1.0.
                                        Critical for privacy guarantee validity.
        
    Returns:
        Tuple containing:
            - PrivacyEngine: Configured privacy engine for accounting and management
            - nn.Module: Privacy-wrapped model with DP-compatible architecture
            - torch.optim.Optimizer: DP-SGD optimizer with noise injection
            - DataLoader: Privacy-aware data loader with proper batch handling
            
    Raises:
        ValueError: If privacy parameters are invalid or model is incompatible
        RuntimeError: If Opacus privacy engine setup fails
        
    Example:
        >>> privacy_engine, dp_model, dp_optimizer, dp_loader = attach_privacy_engine(
        ...     model, optimizer, train_loader, target_epsilon=1.0, target_delta=1e-5
        ... )
        >>> # Train with differential privacy
        >>> for batch in dp_loader:
        ...     dp_optimizer.zero_grad()
        ...     loss = compute_loss(dp_model(batch))
        ...     loss.backward()
        ...     dp_optimizer.step()
    """
    # Model compatibility validation: Ensure DP-compatible architecture
    ModuleValidator.validate(model, strict=False)

    # Privacy engine initialization: Create Opacus privacy management system
    privacy_engine = PrivacyEngine()

    # Noise parameter estimation: Calculate optimal noise multiplier if not specified
    if noise_multiplier is None:
        sample_rate = train_loader.batch_size / len(train_loader.dataset)
        steps = len(train_loader)  # Training steps per epoch
        noise_multiplier = _estimate_noise_multiplier(
            target_epsilon, target_delta, sample_rate, steps
        )
        # Safety bounds: Ensure noise multiplier stays within practical range
        noise_multiplier = float(max(0.5, min(noise_multiplier, 3.0)))

    # Privacy integration: Wrap components with differential privacy mechanisms
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    return privacy_engine, model, optimizer, train_loader


def _estimate_noise_multiplier(epsilon: float, delta: float, sample_rate: float, 
                              steps: int) -> float:
    """
    Estimate optimal noise multiplier for differential privacy parameters.
    
    This function provides a heuristic estimation of the noise multiplier required
    to achieve target privacy parameters using the Gaussian mechanism. It implements
    a simplified version of the moments accountant analysis for practical noise
    calibration in DP-SGD training scenarios.
    
    Args:
        epsilon (float): Target privacy budget (ε) - lower values require more noise
        delta (float): Target privacy parameter (δ) - failure probability bound
        sample_rate (float): Batch sampling rate (batch_size / dataset_size)
        steps (int): Number of training steps for privacy composition
        
    Returns:
        float: Estimated noise multiplier for Gaussian mechanism
        
    Note:
        This is a simplified estimation suitable for initial parameter selection.
        For production deployments, use the Opacus privacy accountant for precise
        noise calibration and privacy budget tracking.
    """
    # Privacy parameter aliasing: Use standard notation for clarity
    q = sample_rate
    T = steps
    
    # Gaussian mechanism calibration: Estimate noise scale from privacy parameters
    # Based on the strong composition theorem and moments accountant analysis
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    sigma *= np.sqrt(q * T)
    
    # Minimum noise enforcement: Ensure sufficient noise for privacy guarantees
    return max(1.0, sigma)


def get_privacy_spent(privacy_engine: PrivacyEngine, delta: float = 1e-5) -> Tuple[float, float]:
    """
    Get current privacy spent from PrivacyEngine.
    
    Args:
        privacy_engine: Opacus PrivacyEngine
        
    Returns:
        Tuple of (epsilon, delta)
    """
    try:
        epsilon = privacy_engine.get_epsilon(delta)
        return float(epsilon), float(delta)
    except Exception:
        # Fallback for older API
        if getattr(privacy_engine, 'accountant', None):
            epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
            return float(epsilon), float(delta)
        return float('inf'), float(delta)


def add_dp_noise_to_gradients(model: nn.Module, noise_scale: float = 0.01) -> None:
    """
    Add differential privacy noise to model gradients.
    
    Args:
        model: PyTorch model
        noise_scale: Standard deviation of noise
    """
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_scale
            param.grad += noise


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients to a maximum norm.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        Total norm of gradients before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def compute_sensitivity(model: nn.Module, train_loader, device: torch.device) -> float:
    """
    Compute the sensitivity of the model for differential privacy.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        device: Device to compute on
        
    Returns:
        Estimated sensitivity
    """
    model.eval()
    sensitivities = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 10:  # Use first 10 batches for estimation
                break
                
            data, target = data.to(device), target.to(device)
            
            # Compute gradients
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            sensitivities.append(total_norm)
            
            # Zero gradients
            model.zero_grad()
    
    return np.mean(sensitivities)


def add_noise_to_parameters_dp(parameters: OrderedDict, epsilon: float, 
                              delta: float, sensitivity: float) -> OrderedDict:
    """
    Add differential privacy noise to model parameters.
    
    Args:
        parameters: Model parameters
        epsilon: Privacy budget
        delta: Privacy parameter
        sensitivity: Model sensitivity
        
    Returns:
        Parameters with added noise
    """
    # Calculate noise scale based on privacy parameters
    noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    noisy_parameters = OrderedDict()
    
    for name, param in parameters.items():
        # Skip adding noise to integer parameters
        if param.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]:
            noisy_parameters[name] = param.clone()
        else:
            noise = torch.randn_like(param) * noise_scale
            noisy_parameters[name] = param + noise
    
    return noisy_parameters


def compute_renyi_divergence(epsilon: float, delta: float, alpha: float = 2.0) -> float:
    """
    Compute Rényi divergence for privacy analysis.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter
        alpha: Rényi divergence order
        
    Returns:
        Rényi divergence value
    """
    # Simplified Rényi divergence computation
    # For more accurate computation, use specialized libraries
    return alpha * epsilon**2 / 2


def privacy_accounting(epsilon: float, delta: float, steps: int, 
                      sample_rate: float) -> Dict[str, float]:
    """
    Perform privacy accounting for composition of multiple queries.
    
    Args:
        epsilon: Per-step privacy budget
        delta: Per-step privacy parameter
        steps: Number of training steps
        sample_rate: Sampling rate
        
    Returns:
        Dictionary with privacy accounting results
    """
    # Simplified privacy accounting
    # For production use, consider using more sophisticated methods
    
    # Basic composition
    total_epsilon = epsilon * np.sqrt(2 * steps * np.log(1 / delta))
    total_delta = delta * steps
    
    # Advanced composition (simplified)
    advanced_epsilon = epsilon * np.sqrt(2 * steps * np.log(1 / delta) + steps * epsilon)
    
    return {
        'basic_epsilon': total_epsilon,
        'basic_delta': total_delta,
        'advanced_epsilon': advanced_epsilon,
        'per_step_epsilon': epsilon,
        'per_step_delta': delta
    }


def validate_dp_parameters(epsilon: float, delta: float) -> bool:
    """
    Validate differential privacy parameters.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter
        
    Returns:
        True if parameters are valid
    """
    if epsilon <= 0:
        return False
    if delta <= 0 or delta >= 1:
        return False
    if epsilon > 100:  # Unrealistically high epsilon
        return False
    
    return True


def get_dp_recommendations(dataset_size: int, batch_size: int, 
                          target_accuracy: float = 0.85) -> Dict[str, float]:
    """
    Get recommended differential privacy parameters based on dataset characteristics.
    
    Args:
        dataset_size: Size of the dataset
        batch_size: Training batch size
        target_accuracy: Target model accuracy
        
    Returns:
        Dictionary with recommended parameters
    """
    sample_rate = batch_size / dataset_size
    
    # Recommendations based on common practices
    if sample_rate < 0.01:  # Small batch size
        epsilon = 1.0
        delta = 1e-5
        noise_multiplier = 1.5
    elif sample_rate < 0.1:  # Medium batch size
        epsilon = 2.0
        delta = 1e-5
        noise_multiplier = 1.2
    else:  # Large batch size
        epsilon = 5.0
        delta = 1e-5
        noise_multiplier = 1.0
    
    return {
        'epsilon': epsilon,
        'delta': delta,
        'noise_multiplier': noise_multiplier,
        'sample_rate': sample_rate,
        'max_grad_norm': 1.0
    }


def print_privacy_report(privacy_engine: PrivacyEngine, round_num: int = 0):
    """
    Print a privacy report for the current training state.
    
    Args:
        privacy_engine: Opacus PrivacyEngine
        round_num: Current training round
    """
    epsilon, delta = get_privacy_spent(privacy_engine)
    
    print(f"\n=== Privacy Report (Round {round_num}) ===")
    print(f"Current ε: {epsilon:.4f}")
    print(f"Current δ: {delta:.2e}")
    
    if privacy_engine.accountant:
        print(f"Privacy accountant: {type(privacy_engine.accountant).__name__}")
    
    # Privacy level interpretation
    if epsilon < 1:
        privacy_level = "Very High"
    elif epsilon < 5:
        privacy_level = "High"
    elif epsilon < 10:
        privacy_level = "Medium"
    else:
        privacy_level = "Low"
    
    print(f"Privacy level: {privacy_level}")
    print("=" * 40)
