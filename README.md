# Circuit Disconnect Duration Prediction

A machine learning system to predict the duration of circuit disconnections based on circuit characteristics and patterns.

## Project Structure

```
circuit_prediction/
│
├── main.py                  # Main execution script
├── data_processing.py       # Data preprocessing and feature engineering
├── ml_models.py             # Machine learning models (RF, XGBoost, etc.)
├── dl_models.py             # Deep learning models (PyTorch)
├── utils.py                 # Utility functions and helper methods
├── README.md                # Project documentation
│
├── results/                 # Directory for saving results
│   ├── models/              # Saved model files
│   ├── plots/               # Generated visualizations
│   └── predictions/         # Prediction outputs
│
└── data/                    # Data directory (if needed)
    └── raw/                 # Raw data files
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/vzb-business-transformation/model_enhancement.git 
cd circuit_prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train models on your data:

```bash
python main.py --data_file path/to/your/data.csv --mode train --models all

# To run only machine learning models
python main.py --mode train --output_dir results --model_type ml

# To run only deep learning models
python main.py --mode train --output_dir results --model_type dl

# To run both (default behavior)
python main.py --mode train --output_dir results --model_type all
```

Options:
- `--data_file`: Path to your input data file (CSV, Excel, Parquet)
- `--mode`: Operation mode (train, predict, evaluate)
- `--target`: Target column for prediction (default: 'DISCO_DURATION')
- `--models`: Models to train (comma-separated or 'all'): rf,gb,xgb,lgb,cat,nn
- `--nn_type`: Neural network type (mlp, lstm, cnn, transformer)
- `--output_dir`: Directory for results (default: 'results')
- `--random_state`: Random seed for reproducibility (default: 42)

### Prediction

To make predictions on new data:

```bash
python main.py --data_file path/to/new_data.csv --mode predict
```

### Evaluation

To evaluate models on data with known targets:

```bash
python main.py --data_file path/to/evaluation_data.csv --mode evaluate
```

## Model Details

### Machine Learning Models

- Random Forest Regressor: Ensemble of decision trees with bagging
- Gradient Boosting Regressor: Sequential ensemble of decision trees
- XGBoost: Optimized gradient boosting implementation
- LightGBM: Light Gradient Boosting Machine
- CatBoost: Gradient boosting with better handling of categorical features
- Stacked Ensemble: Meta-model that combines predictions from multiple base models
- Voting Ensemble: Weighted average of predictions from multiple models

### Deep Learning Models

- MLP (Multi-Layer Perceptron): Standard feed-forward neural network
- LSTM (Long Short-Term Memory): Recurrent neural network for sequence patterns
- CNN (Convolutional Neural Network): Network with convolutional layers for feature extraction
- Transformer: Attention-based architecture for capturing complex dependencies

## Feature Engineering

The system performs advanced feature engineering:

- Time-based features: Cyclic encoding of temporal patterns
- Geographic features: Distance calculations using coordinates
- Speed and disconnect patterns: Relationships between circuit properties
- Customer behavior indicators: Aggregated statistics and patterns
- Interaction features: Combined effects of multiple attributes

## Performance Evaluation

Models are evaluated using multiple metrics:

- R² (Coefficient of Determination): Proportion of variance explained by the model
- RMSE (Root Mean Square Error): Square root of the average squared prediction error
- MAE (Mean Absolute Error): Average absolute prediction error
- MSE (Mean Squared Error): Average squared prediction error

## Requirements

Required packages:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- torch
- matplotlib
- seaborn
- joblib
- category_encoders

# Proxy settings
PROXY_HOST = tpaproxy.verizon.com
PROXY_PORT = 80
PROXY_URL = http://$(PROXY_HOST):$(PROXY_PORT)

.PHONY: all setup install run clean flask

all: install

# Set up conda environment
setup:
	@echo "Creating conda environment..."
	@conda create -n $(CONDA_ENV_NAME) python=3.9 -y
	@echo "Configuring proxy settings..."
	@conda create -n $(CONDA_ENV_NAME) python=3.9 -y
	@conda config --set proxy_servers.http $(PROXY_URL)
	@conda config --set proxy_servers.https $(PROXY_URL)

