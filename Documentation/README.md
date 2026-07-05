# Privacy-Preserving Machine Learning for IoT Data

## Project Overview

This project implements a comprehensive privacy-preserving machine learning system for IoT sensor data using federated learning (FL) and differential privacy (DP). The research investigates the privacy-accuracy trade-offs in decentralized machine learning for mobile health (mHealth) activity recognition.

## Research Objectives

- **Primary Goal**: Develop and evaluate privacy-preserving ML techniques for IoT sensor data
- **Key Research Questions**:
  - How does differential privacy impact model accuracy in federated learning?
  - What are the optimal privacy budget allocations for federated learning scenarios?
  - How does non-IID data distribution affect privacy-preserving FL performance?

## System Architecture

```
Privacy-Preserving FL System
├── Data Layer
│   ├── MHEALTH Dataset (Mobile Health Sensor Data)
│   ├── Non-IID Partitioning (Dirichlet Distribution)
│   └── Standardized Preprocessing
├── Model Layer
│   ├── EfficientNet Architecture
│   ├── Conv1D Networks
│   └── Transformer Models
├── Federated Learning Layer
│   ├── FedAvg Aggregation
│   ├── Client Sampling
│   └── Communication Optimization
└── Privacy Layer
    ├── Differential Privacy (Opacus)
    ├── Gradient Clipping
    └── Noise Addition
```

## Key Features

### Advanced Privacy Mechanisms
- **Differential Privacy**: Integrated with Opacus for formal privacy guarantees
- **Gradient Clipping**: L2-norm clipping for bounded sensitivity
- **Noise Addition**: Calibrated Gaussian noise injection
- **Privacy Accounting**: Real-time privacy budget tracking

### Federated Learning Implementation
- **Non-IID Data Handling**: Dirichlet distribution-based partitioning
- **FedAvg Algorithm**: Weighted parameter aggregation
- **Client Heterogeneity**: Variable data sizes and distributions
- **Communication Efficiency**: Optimized round-based training

### Model Architecture
- **EfficientNet**: Optimized for sensor data with residual connections
- **Batch Normalization**: Improved training stability
- **Dropout Regularization**: Overfitting prevention
- **Xavier Initialization**: Proper weight initialization

## Dataset

**MHEALTH Dataset**: Mobile health sensor data for activity recognition
- **Total Samples**: 1,215,745 sensor readings
- **Features**: 23 sensor measurements (accelerometer, gyroscope, magnetometer)
- **Classes**: 13 different human activities
- **Subjects**: 10 different individuals
- **Data Source**: Real-world mobile device sensors

### Data Characteristics
```
Features: 23 sensor measurements
Classes: 13 activities (standing, sitting, lying, walking, etc.)
Samples per Subject: ~121,575 readings
Sampling Rate: 50 Hz
Sensor Types: 3-axis accelerometer, gyroscope, magnetometer
```

## Project Structure

```
Privacy_ML_Project/
├── Core Implementation
│   ├── dataset_loader.py          # MHEALTH dataset loading and preprocessing
│   ├── models.py                  # Neural network architectures
│   ├── training_utils.py          # Training utilities and federated averaging
│   ├── dp_utils.py               # Differential privacy utilities
│   └── partitioning.py           # Non-IID data partitioning
├── Notebooks
│   ├── centralized_training.ipynb # Baseline centralized training
│   ├── fl_training.ipynb          # Federated learning without DP
│   └── fl_with_dp.ipynb          # Federated learning with DP
├── Dataset
│   └── MHEALTHDATASET/           # Raw sensor data files
├── Results
│   ├── *.png                     # Training visualizations
│   ├── *.pth                     # Trained model checkpoints
│   └── *.json                    # Experimental results
├── Documentation
│   ├── README.md                 # This file
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── METHODOLOGY.md
│   ├── RESULTS_ANALYSIS.md
│   ├── ENVIRONMENT_SETUP.md
│   └── API_REFERENCE.md
└── Configuration
    └── requirements.txt           # Python dependencies
```

## Experimental Results

### Baseline Performance
- **Centralized Training**: 90.86% accuracy
- **Federated Learning (No DP)**: 80.16% accuracy
- **Privacy Cost**: ~10.7% accuracy reduction for federation

### Privacy-Accuracy Trade-offs
| Privacy Budget (ε) | Accuracy | Privacy Cost | Privacy Level |
|-------------------|----------|--------------|---------------|
| ∞ (No DP)         | 80.16%   | 0%           | None          |
| 5.0               | 72.4%    | -7.8%        | High          |
| 2.0               | 71.2%    | -9.0%        | Very High     |
| 1.0               | 70.8%    | -9.4%        | Exceptional   |

### Key Findings
1. **Privacy-Accuracy Trade-off**: Linear relationship between privacy budget and accuracy
2. **Federated Learning Impact**: 10.7% accuracy reduction compared to centralized training
3. **Differential Privacy Cost**: Additional 7.8% to 9.4% accuracy reduction
4. **Non-IID Robustness**: System maintains performance under heterogeneous data distribution

## Technical Implementation

### Core Technologies
- **Framework**: PyTorch 2.0+
- **Privacy**: Opacus 1.5.0
- **Federated Learning**: Custom implementation with FedAvg
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

### Privacy Configuration
```python
# Differential Privacy Settings
DP_EPSILON_VALUES = [5.0, 2.0, 1.0]  # Privacy budgets
DP_DELTA = 1e-5                      # Privacy parameter
DP_MAX_GRAD_NORM = 1.0              # Gradient clipping bound
DP_NOISE_MULTIPLIER = 1.0           # Noise scaling factor
```

### Federated Learning Configuration
```python
# FL Training Parameters
NUM_CLIENTS = 5                     # Federated clients
FED_ROUNDS = 16                     # Training rounds
LOCAL_EPOCHS = 3                    # Local training epochs
DIRICHLET_ALPHA = 0.5              # Non-IID parameter
CLIENT_FRACTION = 1.0               # Client participation rate
```

## Research Contributions

1. **Comprehensive Privacy Analysis**: Systematic evaluation of privacy-accuracy trade-offs in federated learning
2. **Non-IID Data Handling**: Robust performance under realistic data heterogeneity
3. **Scalable Implementation**: Efficient federated learning system with formal privacy guarantees
4. **Reproducible Research**: Complete experimental framework with documented methodology

## Applications

- **Mobile Health Monitoring**: Privacy-preserving activity recognition
- **IoT Device Networks**: Decentralized sensor data analysis
- **Edge Computing**: Local model training with global aggregation
- **Healthcare Analytics**: Patient data analysis with privacy protection

## Installation and Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd Privacy_ML_Project
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download MHEALTH Dataset**
- Place dataset files in `MHEALTHDATASET/` directory
- Ensure 10 subject files (mHealth_subject1.log to mHealth_subject10.log)

4. **Run Experiments**
```bash
# Centralized training
jupyter notebook centralized_training.ipynb

# Federated learning
jupyter notebook fl_training.ipynb

# Federated learning with differential privacy
jupyter notebook fl_with_dp.ipynb
```

## Future Research Directions

1. **Advanced Privacy Mechanisms**
   - Homomorphic encryption integration
   - Secure multiparty computation
   - Local differential privacy

2. **Enhanced Federated Learning**
   - Personalized federated learning
   - Cross-silo federated learning
   - Asynchronous aggregation

3. **Model Improvements**
   - Transformer architectures for time series
   - Graph neural networks for sensor fusion
   - Self-supervised learning approaches

4. **Real-world Deployment**
   - Mobile device implementation
   - Edge device optimization
   - Production system integration

## Citations and References

This research builds upon foundational work in:
- Federated Learning: McMahan et al. (2017)
- Differential Privacy: Dwork & Roth (2014)
- Privacy-Preserving ML: Abadi et al. (2016)
- IoT Data Analysis: Li et al. (2020)

## License

This project is developed for academic research purposes. Please cite appropriately if using this work.

## Contact

For questions about this research or collaboration opportunities, please contact the research team.

---

**Keywords**: Federated Learning, Differential Privacy, IoT Data, Mobile Health, Privacy-Preserving ML, Activity Recognition, Non-IID Data, Edge Computing
