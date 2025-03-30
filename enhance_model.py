# enhance_model.py
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import argparse

# Import your existing modules
from data_processing import DataProcessor
from ml_models import MLModels
from model_enhancer import ModelEnhancer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Circuit Prediction Model Enhancement')

    # parser.add_argument('--jan_predictions', type=str, required=True,
    #                     help='Path to January predictions file')
    #
    # parser.add_argument('--jan_actuals', type=str, required=True,
    #                     help='Path to January actuals data')

    parser.add_argument('--jan_data', type=str, required=True,
                        help='Path to January data for prediction')

    parser.add_argument('--feb_data', type=str, required=True,
                        help='Path to February data for prediction')

    parser.add_argument('--original_model_path', type=str, default='results/models',
                        help='Path to original model directory')

    parser.add_argument('--output_dir', type=str, default='results/enhanced',
                        help='Directory for enhanced results')

    parser.add_argument('--target_col', type=str, default='DISCO_DURATION',
                        help='Target column name')

    parser.add_argument('--pred_col', type=str, default='PREDICTION',
                        help='prediction column in january')

    parser.add_argument('--train_data', type=str, default=None,
                        help='Path to original training data (optional, for retraining)')

    parser.add_argument('--error_threshold', type=float, default=0.3,
                        help='Error threshold for analysis (as percentage)')

    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def load_data(file_path):
    """Load data from various file formats"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def main():
    """Main function to enhance the prediction model"""
    print("\n=== Starting Circuit Prediction Model Enhancement ===\n")
    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load January predictions
    print(f"Loading January predictions from {args.jan_data}")
    jan_data = load_data(args.jan_data)

    required_cols = ['CIR_ID', args.target_col, args.pred_col]
    missing_cols = [col for col in required_cols if col not in jan_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Jan data: {missing_cols}")

    # # Load January actuals
    # print(f"Loading January actuals from {args.jan_actuals}")
    # jan_actuals = load_data(args.jan_actuals)

    # Load February data
    print(f"Loading February data from {args.feb_data}")
    feb_data = load_data(args.feb_data)

    # Initialize components
    data_processor = DataProcessor(random_state=args.random_state)
    ml_models = MLModels(random_state=args.random_state)

    # Load existing models and preprocessor
    data_processor.load_preprocessor(os.path.join(args.original_model_path, 'data_processor.joblib'))
    ml_models.load_models(args.original_model_path)

    # Initialize the model enhancer
    enhancer = ModelEnhancer(
        data_processor=data_processor,
        ml_models=ml_models,
        original_model_path=args.original_model_path,
        output_dir=args.output_dir,
        random_state=args.random_state
    )


    jan_predictions = pd.DataFrame({
        'CIR_ID': jan_data['CIR_ID'],
        'prediction_model': jan_data[args.pred_col]
    })


    # Extract actual values from January data
    jan_actual_values = jan_data[args.target_col]

    feature_cols = [col for col in jan_data.columns if col not in [args.pred_col, 'CIR_ID']]
    jan_features = jan_data[feature_cols]
    # Get features used for predictions
    # jan_features = data_processor.preprocess_data(jan_actuals, is_training=False, target_col=args.target_col)

    # Analyze prediction errors
    error_insights = enhancer.analyze_errors(
        jan_predictions=jan_predictions,
        jan_actuals=jan_actual_values,
        features_df=jan_features,
        threshold=args.error_threshold
    )

    # Load original training data if provided
    if args.train_data:
        print(f"Loading original training data from {args.train_data}")
        train_data = load_data(args.train_data)

        # Retrain with January feedback
        model_results = enhancer.retrain_with_feedback(
            train_data=train_data,
            jan_data=jan_data,
            target_col=args.target_col
        )
    else:
        # If no training data provided, use only error analysis for enhancement
        print("No training data provided. Skipping retraining.")
        model_results = enhancer.retrain_with_feedback(
            train_data=jan_data,
            jan_data=jan_data,
            target_col=args.target_col
        )

    # Generate February predictions
    feb_predictions = enhancer.predict_future(
        feb_data=feb_data,
        target_col=args.target_col
    )

    # Save enhancement report
    enhancer.save_enhancement_report(
        error_analysis=error_insights,
        model_results=model_results,
        start_time=start_time
    )

    print(f"\nModel enhancement completed in {time.time() - start_time:.2f} seconds")
    print(f"Enhanced model and predictions saved to {args.output_dir}")


if __name__ == '__main__':
    main()
