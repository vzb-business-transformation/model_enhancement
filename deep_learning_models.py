import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


class DLModels:
    def __init__(self, input_dim, random_state=42):
        """
        Initialize deep learning models for circuit prediction.

        Args:
            input_dim (int): The dimension of the input features
            random_state (int): Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    def build_lstm(self, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        Create an LSTM model for time series or sequential data.

        Args:
            hidden_dim (int): Size of hidden state
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate

        Returns:
            LSTMModel: PyTorch LSTM model
        """

        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                # Reshape input for sequence: [batch, features] -> [batch, seq_len=1, features]
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)

                lstm_out, _ = self.lstm(x)
                lstm_out = self.dropout(lstm_out[:, -1, :])  # Take the last time step
                return self.fc(lstm_out)

        model = LSTMModel(self.input_dim, hidden_dim, num_layers, dropout)
        model.to(self.device)
        return model

    def build_cnn(self, num_filters=64, kernel_size=3, num_layers=2):
        """
        Create a CNN model for feature extraction.

        Args:
            num_filters (int): Number of convolutional filters
            kernel_size (int): Size of convolutional kernel
            num_layers (int): Number of convolutional layers

        Returns:
            CNNModel: PyTorch CNN model
        """

        class CNNModel(nn.Module):
            def __init__(self, input_dim, num_filters, kernel_size, num_layers):
                super(CNNModel, self).__init__()

                # For 1D convolution, input shape should be [batch, channels, length]
                # So we'll use input_dim as channels

                layers = []
                # First convolutional layer
                layers.append(nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2))
                layers.append(nn.ReLU())

                # Additional convolutional layers
                for _ in range(num_layers - 1):
                    layers.append(
                        nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size // 2))
                    layers.append(nn.ReLU())

                self.conv_layers = nn.Sequential(*layers)

                # Final prediction layer
                self.fc = nn.Linear(num_filters * input_dim, 1)

            def forward(self, x):
                # Reshape input for 1D convolution: [batch, features] -> [batch, channels=1, features]
                x = x.unsqueeze(1)

                # Apply convolutions
                x = self.conv_layers(x)

                # Flatten and apply fully connected layer
                x = x.flatten(1)
                return self.fc(x)

        model = CNNModel(self.input_dim, num_filters, kernel_size, num_layers)
        model.to(self.device)
        return model

    def build_mlp(self, hidden_dims=[128, 64, 32], dropout=0.3):
        """
        Create a Multi-Layer Perceptron (MLP) model.

        Args:
            hidden_dims (list): List of hidden layer dimensions
            dropout (float): Dropout rate

        Returns:
            MLPModel: PyTorch MLP model
        """

        class MLPModel(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout):
                super(MLPModel, self).__init__()

                layers = []
                prev_dim = input_dim

                # Build hidden layers
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim

                # Output layer
                layers.append(nn.Linear(prev_dim, 1))

                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        model = MLPModel(self.input_dim, hidden_dims, dropout)
        model.to(self.device)
        return model

    def build_transformer(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        Create a Transformer model for capturing complex dependencies.

        Args:
            d_model (int): Dimension of the model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout rate

        Returns:
            TransformerModel: PyTorch Transformer model
        """

        class TransformerModel(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
                super(TransformerModel, self).__init__()

                # Input projection
                self.input_proj = nn.Linear(input_dim, d_model)

                # Positional encoding is less relevant for non-sequential data, but included for completeness
                self.pos_encoder = nn.Dropout(dropout)

                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                # Output layer
                self.output_layer = nn.Linear(d_model, 1)

            def forward(self, x):
                # Project input to d_model dimensions
                x = self.input_proj(x)

                # Add fake sequence dimension if not present
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)

                # Apply positional encoding
                x = self.pos_encoder(x)

                # Apply transformer encoder
                x = self.transformer_encoder(x)

                # Take mean across sequence dimension
                x = x.mean(dim=1)

                # Apply output layer
                return self.output_layer(x)

        model = TransformerModel(self.input_dim, d_model, nhead, num_layers, dropout)
        model.to(self.device)
        return model

    def train_model(self, model, X_train, y_train, X_val=None, y_val=None,
                    epochs=100, batch_size=32, learning_rate=0.001,
                    patience=10, verbose=True, max_time_seconds=3600):  # 1 hour default timeout
        """
        Train a PyTorch model with timeout functionality.

        Args:
            model: PyTorch model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            verbose: Whether to print progress
            max_time_seconds: Maximum training time in seconds

        Returns:
            Trained model and training history
        """
        # Record start time for timeout
        start_time = time.time()

        # Convert data to tensors
        print(f"Converting data to tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(self.device)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1).to(self.device)
            has_validation = True
        else:
            has_validation = False

        # Create optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Set up early stopping
        best_val_loss = float('inf')
        best_model = None
        early_stop_counter = 0

        # For recording history
        history = {
            'train_loss': [],
            'val_loss': [] if has_validation else None
        }

        # Training loop
        print(f"Starting training loop...")
        for epoch in range(epochs):
            # Check if timeout has been reached
            if time.time() - start_time > max_time_seconds:
                print(f"Training timed out after {max_time_seconds} seconds")
                break

            model.train()

            # Create mini-batches
            permutation = torch.randperm(X_train_tensor.size(0))
            train_loss = 0.0

            # Add more detailed progress information
            num_batches = (X_train_tensor.size(0) + batch_size - 1) // batch_size

            for i in range(0, X_train_tensor.size(0), batch_size):
                # Check if timeout has been reached (also check within epoch)
                if time.time() - start_time > max_time_seconds:
                    print(f"Training timed out during epoch {epoch + 1}")
                    break

                # Get batch indices
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

                # Print progress for large datasets
                if verbose and i % (10 * batch_size) == 0:
                    batch_num = i // batch_size + 1
                    print(f"  Epoch {epoch + 1}/{epochs}, Batch {batch_num}/{num_batches} "
                          f"({time.time() - start_time:.1f}s elapsed)")

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            # Calculate average train loss
            train_loss /= X_train_tensor.size(0)
            history['train_loss'].append(train_loss)

            # Validation
            if has_validation:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    history['val_loss'].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict().copy()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if verbose:
                    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                          f'Time: {time.time() - start_time:.1f}s')

                # Check for early stopping
                if early_stop_counter >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch + 1}')
                    break
            else:
                if verbose:
                    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, '
                          f'Time: {time.time() - start_time:.1f}s')

        # Load best model if available
        if has_validation and best_model is not None:
            model.load_state_dict(best_model)

        # Final stats
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return model, history
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a trained PyTorch model.

        Args:
            model: Trained PyTorch model
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy().flatten()

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

        return metrics, y_pred

    def plot_training_history(self, history):
        """
        Plot training and validation loss curves.

        Args:
            history: Training history dictionary
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')

        if history['val_loss'] is not None:
            plt.plot(history['val_loss'], label='Validation Loss')

        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

    def process_data_for_lstm(self, X, sequence_length=10):
        """
        Reshape data for LSTM (create sequences).

        Args:
            X: Input features
            sequence_length: Length of sequences to create

        Returns:
            Reshaped data for LSTM input
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # For non-time series data, we can create artificial sequences
        # by repeating each sample sequence_length times
        X_lstm = np.zeros((n_samples, sequence_length, n_features))

        for i in range(n_samples):
            for j in range(sequence_length):
                X_lstm[i, j, :] = X[i, :]

        return X_lstm

    def save_model(self, model, path):
        """
        Save trained PyTorch model.

        Args:
            model: Trained PyTorch model
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': self.input_dim,
            'random_state': self.random_state,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, model_class, path):
        """
        Load a trained PyTorch model.

        Args:
            model_class: The model class to load (e.g., self.build_mlp)
            path: Path to the saved model

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Create a new model instance
        model = model_class()

        # Load the state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"Model loaded from {path}")
        return model


def train_circuit_nn_model(X_train, X_test, y_train, y_test, model_type='mlp'):
    """
    Train a neural network model for circuit disconnect duration prediction.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        model_type: Type of model to train ('mlp', 'lstm', 'cnn', 'transformer')

    Returns:
        Trained model, metrics, and predictions
    """
    # Initialize DL models class
    input_dim = X_train.shape[1]
    dl_models = DLModels(input_dim=input_dim)

    # Create train/validation split
    from sklearn.model_selection import train_test_split
    X_train_nn, X_val, y_train_nn, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Build model based on type
    if model_type == 'lstm':
        model = dl_models.build_lstm(hidden_dim=64, num_layers=2)
        # Reshape data for LSTM if needed
        X_train_nn = dl_models.process_data_for_lstm(X_train_nn)
        X_val = dl_models.process_data_for_lstm(X_val)
        X_test_lstm = dl_models.process_data_for_lstm(X_test)
    elif model_type == 'cnn':
        model = dl_models.build_cnn(num_filters=64, kernel_size=3)
    elif model_type == 'transformer':
        model = dl_models.build_transformer(d_model=64, nhead=4)
    else:  # Default to MLP
        model = dl_models.build_mlp(hidden_dims=[128, 64, 32])

    # Train model
    print(f"Training {model_type.upper()} model...")
    trained_model, history = dl_models.train_model(
        model,
        X_train_nn, y_train_nn,
        X_val, y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=10
    )

    # Plot training history
    dl_models.plot_training_history(history)

    # Evaluate model
    print(f"\nEvaluating {model_type.upper()} model...")
    if model_type == 'lstm':
        metrics, y_pred = dl_models.evaluate_model(trained_model, X_test_lstm, y_test)
    else:
        metrics, y_pred = dl_models.evaluate_model(trained_model, X_test, y_test)

    print(f"{model_type.upper()} Model Performance:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Save model
    model_path = f"results/models/circuit_prediction_{model_type}_model.pt"
    dl_models.save_model(trained_model, model_path)

    return trained_model, metrics, y_pred


if __name__ == "__main__":
    # Example usage
    print("Neural Network models for Circuit Prediction")
    print("This module can be imported and used in the main pipeline")