import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from datetime import datetime

# Import custom modules
from data_processing import DataProcessor
from ml_models import MLModels
from deep_learning_models import DLModels, train_circuit_nn_model



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Circuit Prediction Pipeline')


    parser.add_argument('--data_file', type=str, required=False, default=None,
                            help='Path to input data file (not required when using Teradata)')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'evaluate'],
                        help='Operation mode: train, predict, or evaluate')

    parser.add_argument('--target', type=str, default='DISCO_DURATION',
                        help='Target column for prediction')

    parser.add_argument('--models', type=str, default='all',
                        help='Models to train (comma-separated): rf,gb,xgb,lgb,cat,nn,all')

    parser.add_argument('--nn_type', type=str, default='mlp',
                        choices=['mlp', 'lstm', 'cnn', 'transformer'],
                        help='Type of neural network to use')

    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for results and saved models')

    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--model_type', type=str, default='all',
                        choices=['ml', 'dl', 'all'],
                        help='Type of models to train: ml (Machine Learning only), dl (Deep Learning only), or all')

    return parser.parse_args()


def setup_directories(output_dir):
    """Create output directories if they don't exist"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)


def train_pipeline(args):
    """Train the entire prediction pipeline"""
    print("\n=== Starting Circuit Prediction Training Pipeline ===\n")
    start_time = time.time()

    # 1. Load data

    data_processor = DataProcessor(random_state=args.random_state)
    # Use load_data without arguments when data_file is None
    df = data_processor.load_data() if args.data_file is None else data_processor.load_data(args.data_file)


    # 2. Preprocess data
    processed_df = data_processor.preprocess_data(df, is_training=True, target_col=args.target)

    # 3. Prepare train/test split
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = data_processor.prepare_train_test_data(
        processed_df, target_col=args.target, test_size=0.2
    )

    # 4. Initialize ML models
    ml_models = MLModels(random_state=args.random_state)

    # 5. Feature selection
    X_train_selected = ml_models.select_features(X_train, y_train, threshold=0.001, method='rf')
    X_test_selected = X_test[X_train_selected.columns]

    # 6. Remove highly correlated features
    X_train_final = ml_models.drop_correlated_features(X_train_selected, threshold=0.95)
    X_test_final = X_test_selected[X_train_final.columns]

    # Update scaled data with selected features
    X_train_scaled_final = X_train_scaled[:, X_train.columns.isin(X_train_final.columns)]
    X_test_scaled_final = X_test_scaled[:, X_test.columns.isin(X_train_final.columns)]

    print(f"Final feature set: {X_train_final.shape[1]} features")

    # 7. Determine which models to train
    # models_to_train = []
    # if args.models == 'all':
    #     models_to_train = ['rf', 'gb', 'xgb', 'lgb', 'cat']
    #     train_nn = True
    # else:
    #     model_list = args.models.split(',')
    #     ml_model_types = ['rf', 'gb', 'xgb', 'lgb', 'cat']
    #     models_to_train = [m for m in model_list if m in ml_model_types]
    #     train_nn = 'nn' in model_list

    # Determine which models to train based on model_type
    if args.model_type == 'ml':
        # Only train ML models
        models_to_train = ['rf', 'gb', 'xgb', 'lgb', 'cat']
        train_nn = False
    elif args.model_type == 'dl':
        # Only train DL models
        models_to_train = []
        train_nn = True
    else:  # 'all'
        # Train both ML and DL models
        models_to_train = ['rf', 'gb', 'xgb', 'lgb', 'cat']
        train_nn = True

    # 8. Train ML models
    trained_models = ml_models.train_all_models(X_train_final, y_train, models_to_train=models_to_train)

    # 9. Create ensembles
    if len(trained_models) >= 2:
        stacked_model, stacked_metrics = ml_models.create_stacked_ensemble(
            X_train_final, X_test_final, y_train, y_test
        )

        voting_model, voting_metrics = ml_models.create_voting_ensemble(
            X_train_final, X_test_final, y_train, y_test
        )

    # 10. Train neural network if requested
    if train_nn:
        print("\n=== Training Neural Network ===\n")
        nn_model, nn_metrics, nn_preds = train_circuit_nn_model(
            X_train_final.values, X_test_final.values,
            y_train, y_test,
            model_type=args.nn_type
        )

    # 11. Evaluate all models
    all_results = ml_models.evaluate_models(X_test_final, y_test)

    # 12. Find best model
    best_model_name = max(all_results.items(), key=lambda x: x[1]['R2'])[0]
    best_r2 = all_results[best_model_name]['R2']
    print(f"\nBest model: {best_model_name} with R² = {best_r2:.4f}")

    # 13. Save components
    data_processor.save_preprocessor(os.path.join(args.output_dir, 'models', 'data_processor.joblib'))
    ml_models.save_models(os.path.join(args.output_dir, 'models'))

    # 14. Plot feature importance
    ml_models.plot_feature_importance()
    plt.savefig(os.path.join(args.output_dir, 'plots', 'feature_importance.png'))

    # 15. Save results summary
    save_results_summary(args, all_results, ml_models.feature_importances,
                         X_train_final.shape, start_time)

    print(f"\nTraining pipeline completed in {time.time() - start_time:.2f} seconds")
    print(f"All results saved to {args.output_dir}")


def predict_pipeline(args):
    """Make predictions using trained models"""
    print("\n=== Starting Circuit Prediction Inference Pipeline ===\n")
    start_time = time.time()

    # 1. Load data
    data_processor = DataProcessor(random_state=args.random_state)
    df = data_processor.load_data(args.data_file)

    # 2. Load saved components
    data_processor.load_preprocessor(os.path.join(args.output_dir, 'models', 'data_processor.joblib'))

    ml_models = MLModels(random_state=args.random_state)
    trained_models = ml_models.load_models(os.path.join(args.output_dir, 'models'))

    # 3. Preprocess data
    processed_df = data_processor.preprocess_data(df, is_training=False, target_col=args.target)

    # 4. Make predictions with each model
    predictions = {}
    original_ids = df['CIR_ID'] if 'CIR_ID' in df.columns else pd.Series(range(len(df)))

    for name, model in trained_models.items():
        try:
            preds = model.predict(
                processed_df.drop(columns=[args.target]) if args.target in processed_df.columns else processed_df)
            predictions[name] = preds
        except Exception as e:
            print(f"Error predicting with model {name}: {str(e)}")

    # 5. Ensemble the predictions
    if len(predictions) > 1:
        # Create a simple average ensemble
        ensemble_preds = np.mean([predictions[name] for name in predictions], axis=0)
        predictions['ensemble_average'] = ensemble_preds

    # 6. Create prediction dataframe
    results_df = pd.DataFrame({'CIR_ID': original_ids})

    for name, preds in predictions.items():
        results_df[f'prediction_{name}'] = preds

    # 7. Add actual values if available
    if args.target in df.columns:
        results_df['actual'] = df[args.target]

    # 8. Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, 'predictions', f'predictions_{timestamp}.csv')
    results_df.to_csv(output_file, index=False)

    print(f"\nPrediction pipeline completed in {time.time() - start_time:.2f} seconds")
    print(f"Predictions saved to {output_file}")

    return results_df


def evaluate_pipeline(args):
    """Evaluate models on new data with known targets"""
    print("\n=== Starting Circuit Prediction Evaluation Pipeline ===\n")
    start_time = time.time()

    # 1. Load data
    data_processor = DataProcessor(random_state=args.random_state)
    df = data_processor.load_data()  # Use your Teradata data loading method

    # Ensure target column exists
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in evaluation data")

    # 2. Load saved components
    data_processor.load_preprocessor(os.path.join(args.output_dir, 'models', 'data_processor.joblib'))

    ml_models = MLModels(random_state=args.random_state)
    trained_models = ml_models.load_models(os.path.join(args.output_dir, 'models'))

    # 3. Preprocess data
    processed_df = data_processor.preprocess_data(df, is_training=False, target_col=args.target)

    # 4. Prepare evaluation data
    X_eval = processed_df.drop(columns=[args.target])
    y_eval = processed_df[args.target]

    # 5. Evaluate each model
    results = {}
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_eval)
            metrics = ml_models.calculate_metrics(y_eval, y_pred)
            results[name] = metrics

            print(f"\n{name} model performance:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        except Exception as e:
            print(f"Error evaluating model {name}: {str(e)}")

    print(f"\nEvaluation pipeline completed in {time.time() - start_time:.2f} seconds")

    return results


def main():
    print("Starting main function...")

    # Parse command line arguments
    args = parse_arguments()
    print(f"Arguments parsed: {args}")

    # Setup output directories
    setup_directories(args.output_dir)
    print(f"Directories set up at: {args.output_dir}")

    # Run appropriate pipeline based on mode
    if args.mode == 'train':
        print("Starting training pipeline...")
        train_pipeline(args)
    elif args.mode == 'predict':
        predict_pipeline(args)
    elif args.mode == 'evaluate':
        evaluate_pipeline(args)
    else:
        print(f"Invalid mode: {args.mode}")

    print("Script execution completed.")

if __name__ == '__main__':
    main()