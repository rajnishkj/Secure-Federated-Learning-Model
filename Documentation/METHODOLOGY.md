# Research Methodology

## Research Framework

### Research Questions

This study addresses three primary research questions in the domain of privacy-preserving machine learning for IoT data:

1. **RQ1: Privacy-Accuracy Trade-offs**: How does the application of differential privacy mechanisms impact model accuracy in federated learning scenarios?

2. **RQ2: Optimal Privacy Budget Allocation**: What are the optimal privacy budget (ε) values that balance privacy protection with model utility for IoT sensor data?

3. **RQ3: Non-IID Data Impact**: How does non-identically distributed (non-IID) data across federated clients affect the performance of privacy-preserving machine learning algorithms?

### Research Hypotheses

**H1**: The implementation of differential privacy in federated learning will result in a measurable but acceptable reduction in model accuracy, with stronger privacy guarantees (lower ε values) corresponding to greater accuracy degradation.

**H2**: There exists an optimal privacy budget range (ε ∈ [1.0, 5.0]) that provides meaningful privacy protection while maintaining practical model utility for IoT applications.

**H3**: Non-IID data distribution across federated clients will amplify the accuracy loss introduced by differential privacy mechanisms, but the federated learning protocol will remain robust to moderate levels of data heterogeneity.

## Experimental Design

### Research Methodology

This study employs a **quantitative experimental approach** with controlled comparisons across multiple configurations:

1. **Baseline Establishment**: Centralized training without privacy mechanisms
2. **Federated Learning Baseline**: Federated training without differential privacy
3. **Privacy-Preserving Evaluation**: Federated training with varying differential privacy parameters

### Experimental Variables

#### Independent Variables
- **Privacy Budget (ε)**: [1.0, 2.0, 5.0, ∞ (no DP)]
- **Data Distribution**: Non-IID with Dirichlet parameter α = 0.5
- **Number of Clients**: 5 federated participants
- **Training Configuration**: Consistent across all experiments

#### Dependent Variables
- **Model Accuracy**: Primary performance metric
- **Convergence Rate**: Training rounds to stability
- **Privacy Loss**: Quantified utility degradation
- **Communication Efficiency**: Rounds required for convergence

#### Control Variables
- **Model Architecture**: EfficientNet with consistent hyperparameters
- **Dataset**: MHEALTH with standardized preprocessing
- **Random Seeds**: Fixed for reproducibility
- **Hardware**: Consistent computational environment

### Threat Model

#### Privacy Assumptions
- **Adversary Type**: Honest-but-curious server and clients
- **Attack Model**: Passive inference attacks on model parameters
- **Knowledge**: Adversary has access to aggregated model updates
- **Objective**: Prevent individual data point reconstruction

#### Security Guarantees
- **Differential Privacy**: (ε, δ)-differential privacy with δ = 1e-5
- **Gradient Clipping**: L2-norm bound of 1.0 for sensitivity control
- **Noise Mechanism**: Gaussian noise calibrated to privacy parameters

## Data Collection and Preparation

### Dataset Description

**MHEALTH Dataset**: A comprehensive mobile health dataset containing sensor readings for human activity recognition.

#### Dataset Characteristics
- **Source**: UCI Machine Learning Repository
- **Domain**: Mobile health and activity recognition
- **Collection Method**: Real-world sensor data from mobile devices
- **Temporal Coverage**: Continuous activity monitoring
- **Demographic Diversity**: 10 different subjects

#### Data Specifications
```
Total Samples: 1,215,745 sensor readings
Features: 23 dimensions (3-axis accelerometer, gyroscope, magnetometer)
Classes: 13 activities (standing, sitting, lying, walking, etc.)
Subjects: 10 individuals with varying activity patterns
Sampling Rate: 50 Hz continuous monitoring
File Format: Space-separated values with class labels
```

#### Data Quality Assessment
- **Completeness**: 100% of samples contain all 23 features
- **Consistency**: Standardized sensor reading ranges
- **Validity**: All activity labels verified and consistent
- **Reliability**: Multiple subjects per activity class

### Data Preprocessing Pipeline

#### 1. Data Loading and Validation
```python
Process:
1. Multi-format file reading (space, comma, tab delimited)
2. Feature extraction (columns 1-23)
3. Label extraction and validation (column 24)
4. Missing value detection and handling
5. Data type validation and conversion
```

#### 2. Feature Standardization
```python
Method: Z-score normalization per subject file
Formula: z = (x - μ) / σ
Rationale: 
- Accounts for individual sensor calibration differences
- Maintains relative activity patterns
- Improves neural network convergence
```

#### 3. Data Partitioning Strategy

##### Non-IID Partitioning Implementation
```python
Method: Dirichlet Distribution Sampling
Parameters:
- α = 0.5 (heterogeneity parameter)
- num_clients = 5
- Distribution: Dir(α) for each class across clients

Rationale:
- Simulates realistic federated scenarios
- Controls level of data heterogeneity
- Maintains class representation across clients
```

##### Partition Analysis
```python
Metrics:
- Class distribution entropy per client
- KL-divergence from uniform distribution
- Heterogeneity score: L1 distance from ideal distribution
- Sample size variance across clients
```

## Model Architecture and Training

### Neural Network Design

#### EfficientNet Architecture
```python
Architecture Design:
Input Layer: 23 sensor features
Hidden Layers: [256, 256, 128, 64] neurons
Output Layer: 13 activity classes
Activation: ReLU with batch normalization
Regularization: Dropout (rate: 0.2) + L2 weight decay
Residual Connections: Between layers of same dimension
```

#### Architecture Justification
- **Parameter Efficiency**: Optimized for federated learning communication
- **Convergence Stability**: Batch normalization and residual connections
- **Regularization**: Prevents overfitting on non-IID data
- **DP Compatibility**: Architecture validated with Opacus requirements

### Training Protocol

#### Centralized Training (Baseline)
```python
Configuration:
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Batch Size: 512
- Epochs: 10
- Loss Function: CrossEntropy with label smoothing (0.05)
- Scheduler: Cosine annealing with warmup
```

#### Federated Learning Protocol
```python
FedAvg Implementation:
1. Server initializes global model θ₀
2. For round t = 1 to T:
   a. Sample client subset (100% participation)
   b. Broadcast θₜ to selected clients
   c. Local training: θₜᵢ ← LocalUpdate(θₜ, Dᵢ)
   d. Server aggregation: θₜ₊₁ ← Σᵢ(nᵢ/n)θₜᵢ
3. Return final global model θₜ

Local Training Parameters:
- Local epochs: 3
- Batch size: 512 (memory permitting)
- Learning rate: 0.001 (adaptive based on ε)
- Gradient clipping: max_norm = 1.0
```

### Differential Privacy Implementation

#### Privacy Mechanism Design
```python
DP-SGD Integration:
1. Gradient computation: ∇L(θ, xᵢ)
2. Gradient clipping: clip(∇L, C) where C = 1.0
3. Noise addition: ∇̃L = ∇L + N(0, σ²C²I)
4. Parameter update: θ ← θ - η∇̃L

Noise Calibration:
σ = C√(2ln(1.25/δ)) / ε
Where: C = clipping bound, ε = privacy budget, δ = failure probability
```

#### Privacy Accounting
```python
Composition Method: Rényi Differential Privacy (RDP)
Accounting: Real-time ε consumption tracking
Budget Allocation: Equal distribution across training rounds
Early Stopping: Halt training when budget exhausted
```

## Evaluation Framework

### Performance Metrics

#### Primary Metrics
1. **Classification Accuracy**: Percentage of correctly classified test samples
2. **Privacy Loss**: Accuracy degradation due to privacy mechanisms
3. **Convergence Rate**: Training rounds required for stability

#### Secondary Metrics
1. **F1-Score**: Balanced performance across activity classes
2. **Communication Rounds**: Efficiency of federated protocol
3. **Privacy Budget Consumption**: Actual ε used vs. allocated

### Experimental Protocols

#### Controlled Experiments
```python
Experiment 1: Centralized Baseline
- Training: Complete dataset on single machine
- Objective: Establish maximum achievable accuracy
- Output: Baseline performance benchmark

Experiment 2: Federated Learning Baseline
- Training: Distributed across 5 clients, no privacy
- Objective: Quantify federation cost
- Output: Federation impact on accuracy

Experiment 3: Privacy-Preserving Federated Learning
- Training: Distributed with DP (ε ∈ [1.0, 2.0, 5.0])
- Objective: Evaluate privacy-accuracy trade-offs
- Output: Privacy cost analysis
```

#### Statistical Analysis
```python
Methodology:
- Multiple runs per configuration (n=3)
- Mean and standard deviation reporting
- Confidence intervals (95%)
- Statistical significance testing (t-tests)
```

### Reproducibility Measures

#### Experimental Control
```python
Reproducibility Framework:
- Fixed random seeds: 42 across all experiments
- Version pinning: Exact library versions specified
- Hardware consistency: Same computational environment
- Code documentation: Complete parameter logging
```

#### Data Versioning
```python
Dataset Management:
- Checksum verification of source data
- Preprocessing pipeline documentation
- Partition seed tracking
- Data split preservation
```

## Privacy Analysis Framework

### Formal Privacy Guarantees

#### Differential Privacy Definition
For a mechanism M: D → R, M satisfies (ε, δ)-differential privacy if for all neighboring datasets D and D' differing by one record, and for all possible outputs S ⊆ Range(M):

```
P[M(D) ∈ S] ≤ exp(ε) × P[M(D') ∈ S] + δ
```

#### Privacy Budget Analysis
```python
Budget Allocation Strategy:
- Total Budget: ε_total ∈ [1.0, 2.0, 5.0]
- Per-Round Budget: ε_round = ε_total / num_rounds
- Per-Client Budget: Equal allocation across clients
- Composition: Advanced composition via RDP
```

### Attack Model Evaluation

#### Baseline Attacks
1. **Model Inversion**: Attempt to reconstruct training data from model parameters
2. **Membership Inference**: Determine if specific samples were used in training
3. **Property Inference**: Extract sensitive attributes from model behavior

#### Defense Evaluation
```python
Privacy Evaluation:
- Empirical privacy loss measurement
- Theoretical guarantee verification
- Attack success rate under different ε values
- Utility-privacy trade-off quantification
```

## Quality Assurance and Validation

### Code Quality Standards

#### Development Practices
```python
Standards:
- Modular design with clear separation of concerns
- Comprehensive unit testing for core functions
- Type hints and documentation for all public APIs
- Code review process for critical components
```

#### Testing Framework
```python
Test Coverage:
- Unit tests: Individual function validation
- Integration tests: Component interaction verification
- End-to-end tests: Complete workflow validation
- Performance tests: Resource usage monitoring
```

### Experimental Validation

#### Internal Validation
```python
Validation Checks:
- Model convergence verification
- Privacy accounting accuracy
- Gradient clipping effectiveness
- Noise calibration correctness
```

#### External Validation
```python
Benchmark Comparison:
- Literature baseline comparison
- Standard dataset evaluation
- Open-source implementation verification
- Peer review and feedback incorporation
```

## Limitations and Constraints

### Technical Limitations

#### Hardware Constraints
- **Memory**: Limited by GPU memory for large batch sizes
- **Computation**: Extended training time for DP mechanisms
- **Storage**: Large dataset storage requirements

#### Software Limitations
- **Framework Dependencies**: Opacus compatibility requirements
- **Precision**: Numerical stability with noise addition
- **Scalability**: Current implementation limited to moderate client numbers

### Methodological Constraints

#### Dataset Limitations
- **Domain Specificity**: Results specific to activity recognition
- **Sample Size**: Limited to available MHEALTH data
- **Demographic Bias**: Potential bias in subject population

#### Experimental Scope
- **Privacy Model**: Limited to honest-but-curious adversaries
- **Attack Evaluation**: Basic inference attacks only
- **Real-world Factors**: Simplified communication model

### Generalizability Considerations

#### Domain Transfer
- **IoT Applications**: Results may generalize to similar sensor data
- **Privacy Requirements**: Framework applicable to various privacy needs
- **Scale Variations**: May require adjustment for different client numbers

#### Future Research Directions
- **Advanced Privacy Models**: Stronger adversary assumptions
- **Real-world Deployment**: Practical implementation challenges
- **Cross-domain Evaluation**: Different IoT data types and applications

## Ethical Considerations

### Privacy Protection
- **Formal Guarantees**: Mathematical privacy protection proof
- **Transparency**: Clear documentation of privacy mechanisms
- **User Control**: Configurable privacy levels
- **Data Minimization**: Only necessary data collection

### Research Ethics
- **Public Dataset**: Use of publicly available data only
- **No Human Subjects**: No direct human experimentation
- **Reproducibility**: Complete methodology disclosure
- **Beneficial Impact**: Research aimed at privacy protection improvement
