import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import logging
from datetime import datetime


def setup_logger(log_file=None):
    """
    Set up logging configuration.

    Args:
        log_file (str): Path to log file

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger('circuit_prediction')
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_metrics(y_true, y_pred, prefix=''):
    """
    Calculate regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        prefix (str): Prefix for metric names

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
        f'{prefix}MSE': mean_squared_error(y_true, y_pred),
        f'{prefix}RMSE': mean_squared_error(y_true, y_pred, squared=False),
        f'{prefix}R2': r2_score(y_true, y_pred)
    }

    # Add MAPE if no zeros in true values
    if not np.any(y_true == 0):
        metrics[f'{prefix}MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return metrics


def plot_predictions(y_true, y_pred, title='Actual vs Predicted', save_path=None):
    """
    Plot actual vs predicted values.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Labels and title
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)

    # Add metrics
    metrics = calculate_metrics(y_true, y_pred)
    metrics_text = f"R² = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.4f}\nMAE = {metrics['MAE']:.4f}"
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_residuals(y_true, y_pred, title='Residual Plot', save_path=None):
    """
    Plot residuals.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.scatter(y_pred, residuals, alpha=0.6)

    # Zero line
    plt.axhline(y=0, color='r', linestyle='--')

    # Labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)

    # Add residual statistics
    stats_text = f"Mean = {residuals.mean():.4f}\nStd Dev = {residuals.std():.4f}"
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_feature_importance(importance, feature_names, title='Feature Importance', top_n=20, save_path=None):
    """
    Plot feature importance.

    Args:
        importance: Feature importance values
        feature_names: Feature names
        title (str): Plot title
        top_n (int): Number of top features to show
        save_path (str): Path to save the plot
    """
    # Create Series if not already
    if not isinstance(importance, pd.Series):
        importance = pd.Series(importance, index=feature_names)

    # Sort and get top N
    importance = importance.sort_values(ascending=False)
    top_importance = importance.head(top_n)

    plt.figure(figsize=(12, 8))
    top_importance.plot(kind='barh')
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_learning_curves(train_history, val_history=None, metric='loss', title='Learning Curve', save_path=None):
    """
    Plot learning curves.

    Args:
        train_history: Training metric history
        val_history: Validation metric history
        metric (str): Metric name
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_history) + 1)
    plt.plot(epochs, train_history, 'b-', label=f'Training {metric}')

    if val_history is not None:
        plt.plot(epochs, val_history, 'r-', label=f'Validation {metric}')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix for classification tasks.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        classes: Class names
        normalize (bool): Whether to normalize the matrix
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def regression_report(y_true, y_pred, model_name='Model'):
    """
    Generate a comprehensive regression report.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name (str): Name of the model

    Returns:
        str: Formatted report
    """
    metrics = calculate_metrics(y_true, y_pred)

    report = f"=== {model_name} Regression Report ===\n\n"

    # Add metrics
    report += "Performance Metrics:\n"
    report += f"R² Score: {metrics['R2']:.4f}\n"
    report += f"RMSE: {metrics['RMSE']:.4f}\n"
    report += f"MSE: {metrics['MSE']:.4f}\n"
    report += f"MAE: {metrics['MAE']:.4f}\n"

    if 'MAPE' in metrics:
        report += f"MAPE: {metrics['MAPE']:.2f}%\n"

    # Add distribution statistics
    report += "\nPrediction Statistics:\n"
    report += f"Actual Mean: {y_true.mean():.4f}\n"
    report += f"Predicted Mean: {y_pred.mean():.4f}\n"
    report += f"Actual Std Dev: {y_true.std():.4f}\n"
    report += f"Predicted Std Dev: {y_pred.std():.4f}\n"

    # Add residual statistics
    residuals = y_true - y_pred
    report += "\nResidual Statistics:\n"
    report += f"Mean Residual: {residuals.mean():.4f}\n"
    report += f"Std Dev of Residuals: {residuals.std():.4f}\n"

    # Add percentile information
    report += "\nPercentile Error Analysis:\n"
    for p in [10, 25, 50, 75, 90]:
        err_percentile = np.percentile(np.abs(residuals), p)
        report += f"{p}th Percentile of Absolute Error: {err_percentile:.4f}\n"

    return report


def save_model_artifacts(models, feature_importances, metrics, config, output_dir):
    """
    Save model artifacts and results.

    Args:
        models (dict): Dictionary of trained models
        feature_importances (dict): Dictionary of feature importances
        metrics (dict): Dictionary of model metrics
        config (dict): Configuration parameters
        output_dir (str): Output directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # Save models
    import joblib
    for name, model in models.items():
        model_path = os.path.join(output_dir, 'models', f"{name}_{timestamp}.joblib")
        joblib.dump(model, model_path)

    # Save feature importances
    if feature_importances:
        for name, importance in feature_importances.items():
            # Plot
            plot_path = os.path.join(output_dir, 'plots', f"feature_importance_{name}_{timestamp}.png")
            plot_feature_importance(importance, importance.index,
                                    title=f"{name} Feature Importance", save_path=plot_path)

            # Save data
            importance_path = os.path.join(output_dir, 'results', f"feature_importance_{name}_{timestamp}.csv")
            importance.to_csv(importance_path)

    # Save metrics
    metrics_path = os.path.join(output_dir, 'results', f"metrics_{timestamp}.csv")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(metrics_path)

    # Save configuration
    config_path = os.path.join(output_dir, 'results', f"config_{timestamp}.txt")
    with open(config_path, 'w') as f:
        f.write("=== Configuration ===\n\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"All artifacts saved to {output_dir}")


def load_model_artifacts(model_dir, result_dir=None):
    """
    Load saved model artifacts.

    Args:
        model_dir (str): Directory containing model files
        result_dir (str): Directory containing result files

    Returns:
        tuple: Loaded models and metrics
    """
    # Load models
    import joblib
    models = {}

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    for model_file in model_files:
        name = model_file.split('_')[0]  # Extract model name
        model_path = os.path.join(model_dir, model_file)
        models[name] = joblib.load(model_path)

    # Load metrics if result directory provided
    metrics = None
    if result_dir:
        metrics_files = [f for f in os.listdir(result_dir) if f.startswith('metrics_') and f.endswith('.csv')]

        if metrics_files:
            # Load most recent metrics file
            latest_metrics = sorted(metrics_files)[-1]
            metrics_path = os.path.join(result_dir, latest_metrics)
            metrics = pd.read_csv(metrics_path, index_col=0).to_dict()

    return models, metrics