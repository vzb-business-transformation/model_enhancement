# model_enhancer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any
import joblib
import os
import time
from datetime import datetime


class ModelEnhancer:
    """
    Class to enhance model performance by analyzing errors and incorporating feedback.
    """

    def __init__(self,
                 data_processor=None,
                 ml_models=None,
                 original_model_path=None,
                 output_dir="results/enhanced",
                 random_state=42):
        """
        Initialize the model enhancer.

        Args:
            data_processor: DataProcessor instance
            ml_models: MLModels instance
            original_model_path: Path to original model
            output_dir: Directory for enhanced model outputs
            random_state: Random seed for reproducibility
        """
        self.data_processor = data_processor
        self.ml_models = ml_models
        self.original_model_path = original_model_path
        self.output_dir = output_dir
        self.random_state = random_state
        self.error_patterns = None
        self.feature_importance_shift = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def analyze_errors(self,
                       jan_predictions: pd.DataFrame,
                       jan_actuals: pd.Series,
                       features_df: pd.DataFrame,
                       threshold: float = 0.3) -> Dict:
        """
        Analyze prediction errors to identify patterns.

        Args:
            jan_predictions: DataFrame with model predictions
            jan_actuals: Series with actual values
            features_df: DataFrame with feature values
            threshold: Error threshold for analysis (as percentage)

        Returns:
            Dict of error patterns and insights
        """
        print("\n=== Analyzing Prediction Errors ===\n")

        # Create a DataFrame with predictions and actuals
        results = pd.DataFrame({
            'actual': jan_actuals
        })

        # Add predictions from each model
        if 'prediction_model' in jan_predictions.columns:
            results['pred_model'] = jan_predictions['prediction_model']

            # Calculate errors
            results['error_model'] = results['pred_model'] - results['actual']
            results['abs_error_model'] = results['error_model'].abs()
            results['pct_error_model'] = (results['abs_error_model'] / results['actual'].replace(0,  1)) * 100
            model_names = ['model']

        else:
            model_names = []
            for col in jan_predictions.columns:
                if col.startswith('prediction_'):
                    model_name = col.replace('prediction_', '')
                    model_names.append(model_name)

                    results[f'pred_{model_name}'] = jan_predictions[col]

                    # Calculate errors
                    results[f'error_{model_name}'] = results[f'pred_{model_name}'] - results['actual']
                    results[f'abs_error_{model_name}'] = results[f'error_{model_name}'].abs()
                    results[f'pct_error_{model_name}'] = (results[f'abs_error_{model_name}'] / results['actual'].replace(0,
                                                                                                                     1)) * 100

        # Find high-error cases for each model
        error_insights = {}

        # for model_name in [col.replace('pred_', '') for col in results.columns if col.startswith('pred_')]:
        for model_name in model_names:
            # Identify high error cases
            high_errors = results[results[f'pct_error_{model_name}'] > threshold * 100].copy()

            print(
                f"Model {model_name}: {len(high_errors)} high-error cases out of {len(results)} ({len(high_errors) / len(results) * 100:.2f}%)")

            if len(high_errors) == 0:
                error_insights[model_name] = {
                    'high_error_count': 0,
                    'error_bias': 0,
                    'correlated_features': {},
                    'error_by_value_range': {}
                }
                continue

            # Check for systematic bias in errors
            error_bias = results[f'error_{model_name}'].mean()
            over_predictions = (results[f'error_{model_name}'] > 0).sum()
            under_predictions = (results[f'error_{model_name}'] < 0).sum()

            print(f"  - Bias direction: {'Overprediction' if error_bias > 0 else 'Underprediction'}")
            print(f"  - Bias magnitude: {abs(error_bias):.2f}")
            print(f"  - Over-predictions: {over_predictions} ({over_predictions / len(results) * 100:.2f}%)")
            print(f"  - Under-predictions: {under_predictions} ({under_predictions / len(results) * 100:.2f}%)")

            # Merge errors with features for analysis
            if features_df is not None and len(features_df) == len(results):
                high_error_features = pd.concat([high_errors, features_df.loc[high_errors.index]], axis=1)

                # Find features correlated with errors
                correlations = {}
                numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns

                for feature in numeric_features:
                    if feature in features_df.columns:
                        corr = np.corrcoef(
                            features_df[feature].values,
                            results[f'abs_error_{model_name}'].values
                        )[0, 1]

                        if not np.isnan(corr):
                            correlations[feature] = corr

                # Sort correlations
                sorted_correlations = {k: v for k, v in sorted(correlations.items(),
                                                               key=lambda item: abs(item[1]),
                                                               reverse=True)}

                # Get top correlated features
                top_features = dict(list(sorted_correlations.items())[:10])

                print(f"  - Top features correlated with error:")
                for feature, corr in top_features.items():
                    print(f"    {feature}: {corr:.4f}")

                # Analyze errors by value ranges for top correlated features
                error_by_range = {}

                for feature, corr in list(top_features.items())[:3]:  # Analyze top 3 features
                    try:
                        # Create bins
                        bin_edges = np.quantile(features_df[feature], [0, 0.25, 0.5, 0.75, 1])
                        bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']

                        # Bin the data
                        features_df[f'{feature}_bin'] = pd.cut(
                            features_df[feature],
                            bins=bin_edges,
                            labels=bin_labels,
                            include_lowest=True
                        )

                        # Calculate mean error by bin
                        error_by_bin = results.groupby(features_df[f'{feature}_bin'])[f'abs_error_{model_name}'].mean()
                        error_by_range[feature] = error_by_bin.to_dict()

                        print(f"  - Error by {feature} range:")
                        for bin_name, mean_error in error_by_bin.items():
                            print(f"    {bin_name}: {mean_error:.4f}")

                    except Exception as e:
                        print(f"    Error analyzing {feature} by range: {str(e)}")

                # Store insights
                error_insights[model_name] = {
                    'high_error_count': len(high_errors),
                    'error_bias': error_bias,
                    'over_predictions': over_predictions,
                    'under_predictions': under_predictions,
                    'correlated_features': top_features,
                    'error_by_value_range': error_by_range
                }
            else:
                print("  - Cannot analyze feature correlation: features_df missing or size mismatch")
                error_insights[model_name] = {
                    'high_error_count': len(high_errors),
                    'error_bias': error_bias,
                    'over_predictions': over_predictions,
                    'under_predictions': under_predictions,
                    'correlated_features': {},
                    'error_by_value_range': {}
                }

            # Visualize error distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(results[f'error_{model_name}'], kde=True)
            plt.title(f'Error Distribution for {model_name}')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='r', linestyle='--')
            plt.savefig(os.path.join(self.output_dir, f'error_dist_{model_name}.png'))
            plt.close()

        self.error_patterns = error_insights
        return error_insights

    def add_targeted_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add new features targeting specific error patterns.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with additional features
        """
        print("\n=== Adding Targeted Features ===\n")

        if self.error_patterns is None:
            print("No error patterns available. Run analyze_errors first.")
            return X

        X_enhanced = X.copy()

        # Process each model's error patterns
        for model_name, insights in self.error_patterns.items():
            # Focus on top correlated features
            for feature, corr in insights['correlated_features'].items():
                if feature in X.columns and abs(corr) > 0.1:  # Only process features with meaningful correlation
                    print(f"Adding targeted features for {feature} (correlation: {corr:.4f})")

                    # Create polynomial features
                    X_enhanced[f'{feature}_squared'] = X[feature] ** 2
                    X_enhanced[f'{feature}_cubed'] = X[feature] ** 3

                    # Create log transformation
                    if (X[feature] > 0).all():
                        X_enhanced[f'log_{feature}'] = np.log1p(X[feature])

                    # Create interaction features between top correlated features
                    for other_feature, other_corr in insights['correlated_features'].items():
                        if (other_feature in X.columns and
                                feature != other_feature and
                                abs(other_corr) > 0.1):
                            X_enhanced[f'{feature}_x_{other_feature}'] = X[feature] * X[other_feature]

            # Add correction factor for systematic bias
            if abs(insights['error_bias']) > 5:  # Only add if bias is significant
                bias_direction = 1 if insights['error_bias'] > 0 else -1

                # Calculate adjustment factor based on bias
                X_enhanced[f'bias_correction_{model_name}'] = bias_direction * abs(insights['error_bias']) * 0.5

                print(f"Adding bias correction factor for {model_name}: {insights['error_bias']:.4f}")

        # Check how many new features were added
        new_features = [col for col in X_enhanced.columns if col not in X.columns]
        print(f"Added {len(new_features)} new targeted features")

        return X_enhanced

    def retrain_with_feedback(self,
                              train_data: pd.DataFrame,
                              jan_data: pd.DataFrame,
                              target_col: str = 'DISCO_DURATION') -> Dict:
        """
        Retrain models with January data incorporated.

        Args:
            train_data: Original training data
            jan_data: January data with actual values
            target_col: Target column name

        Returns:
            Dict of enhanced models
        """
        print("\n=== Retraining Models with Feedback ===\n")

        # Combine original training data with January data
        combined_data = pd.concat([train_data, jan_data], ignore_index=True)
        print(f"Combined data shape: {combined_data.shape}")

        # Preprocess combined data
        processed_df = self.data_processor.preprocess_data(combined_data, is_training=True, target_col=target_col)
        print(f"Processed data shape: {processed_df.shape}")

        # Prepare train/test split
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = self.data_processor.prepare_train_test_data(
            processed_df, target_col=target_col, test_size=0.2
        )

        # Feature selection with original method
        X_train_selected = self.ml_models.select_features(X_train, y_train, threshold=0.001, method='rf')
        X_test_selected = X_test[X_train_selected.columns]

        # Add targeted features based on error analysis
        X_train_enhanced = self.add_targeted_features(X_train_selected)
        X_test_enhanced = self.add_targeted_features(X_test_selected)

        # Remove highly correlated features
        X_train_final = self.ml_models.drop_correlated_features(X_train_enhanced, threshold=0.95)
        X_test_final = X_test_enhanced[X_train_final.columns]

        print(f"Final enhanced feature set: {X_train_final.shape[1]} features")

        # Train models with new feature set
        models_to_train = ['rf', 'gb', 'xgb', 'lgb', 'cat']
        enhanced_models = self.ml_models.train_all_models(X_train_final, y_train, models_to_train=models_to_train)

        # Create ensembles
        if len(enhanced_models) >= 2:
            stacked_model, stacked_metrics = self.ml_models.create_stacked_ensemble(
                X_train_final, X_test_final, y_train, y_test
            )

            voting_model, voting_metrics = self.ml_models.create_voting_ensemble(
                X_train_final, X_test_final, y_train, y_test
            )

        # Evaluate all models
        all_results = self.ml_models.evaluate_models(X_test_final, y_test)

        # Find best model
        best_model_name = max(all_results.items(), key=lambda x: x[1]['R2'])[0]
        best_r2 = all_results[best_model_name]['R2']
        print(f"\nBest enhanced model: {best_model_name} with R² = {best_r2:.4f}")

        # Compare feature importance before and after enhancement
        if hasattr(self.ml_models, 'feature_importances') and 'rf' in self.ml_models.feature_importances:
            old_importances = pd.read_joblib(os.path.join(os.path.dirname(self.original_model_path),
                                                          "feature_importances.joblib"))

            if 'rf' in old_importances:
                old_top_features = old_importances['rf'].head(20)
                new_top_features = self.ml_models.feature_importances['rf'].head(20)

                # Find changes in feature importance
                common_features = set(old_top_features.index) & set(new_top_features.index)

                importance_shifts = {}
                for feature in common_features:
                    old_rank = old_top_features.index.get_loc(feature)
                    new_rank = new_top_features.index.get_loc(feature)
                    rank_change = old_rank - new_rank

                    old_importance = old_top_features[feature]
                    new_importance = new_top_features[feature]
                    importance_change = (new_importance - old_importance) / old_importance * 100

                    importance_shifts[feature] = {
                        'old_rank': old_rank,
                        'new_rank': new_rank,
                        'rank_change': rank_change,
                        'importance_change_pct': importance_change
                    }

                self.feature_importance_shift = importance_shifts

                # Print top changes
                print("\nTop feature importance changes:")
                for feature, changes in sorted(importance_shifts.items(),
                                               key=lambda x: abs(x[1]['importance_change_pct']),
                                               reverse=True)[:5]:
                    print(
                        f"  {feature}: {changes['importance_change_pct']:.2f}% change, rank: {changes['old_rank']} → {changes['new_rank']}")

        # Save enhanced models
        self.ml_models.save_models(os.path.join(self.output_dir, 'models'))

        # Save data processor
        self.data_processor.save_preprocessor(os.path.join(self.output_dir, 'models', 'data_processor.joblib'))

        return all_results

    def predict_future(self, feb_data: pd.DataFrame, target_col: str = 'DISCO_DURATION') -> pd.DataFrame:
        """
        Generate predictions for February data.

        Args:
            feb_data: February data for prediction
            target_col: Target column name

        Returns:
            DataFrame with predictions
        """
        print("\n=== Generating February Predictions ===\n")

        # Preprocess February data
        processed_df = self.data_processor.preprocess_data(feb_data, is_training=False, target_col=target_col)

        # Create prediction dataframe
        original_ids = feb_data['CIR_ID'] if 'CIR_ID' in feb_data.columns else pd.Series(range(len(feb_data)))
        results_df = pd.DataFrame({'CIR_ID': original_ids})

        # Add actual values if available
        if target_col in feb_data.columns:
            results_df['actual'] = feb_data[target_col]

        # Add targeted features to processed data
        enhanced_features = self.add_targeted_features(processed_df)

        # Make predictions with each model
        predictions = {}

        for name, model in self.ml_models.best_models.items():
            try:
                if target_col in enhanced_features.columns:
                    X = enhanced_features.drop(columns=[target_col])
                else:
                    X = enhanced_features

                preds = model.predict(X)
                predictions[name] = preds
                results_df[f'prediction_{name}'] = preds
            except Exception as e:
                print(f"Error predicting with model {name}: {str(e)}")

        # Create ensemble prediction (average of all models)
        if len(predictions) > 1:
            ensemble_preds = np.mean([predictions[name] for name in predictions], axis=0)
            results_df['prediction_ensemble'] = ensemble_preds

        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f'feb_predictions_{timestamp}.csv')
        results_df.to_csv(output_file, index=False)

        print(f"February predictions saved to {output_file}")

        return results_df

    def save_enhancement_report(self,
                                error_analysis: Dict,
                                model_results: Dict,
                                start_time: float):
        """
        Save a detailed report of the enhancement process.

        Args:
            error_analysis: Results from error analysis
            model_results: Results from model evaluation
            start_time: Start time for timing
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f'enhancement_report_{timestamp}.txt')

        with open(report_file, 'w') as f:
            f.write("=== Circuit Prediction Model Enhancement Report ===\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("=== Error Analysis Summary ===\n")
            for model_name, insights in error_analysis.items():
                f.write(f"\nModel: {model_name}\n")
                f.write("-" * (len(model_name) + 8) + "\n")
                f.write(f"High-error cases: {insights['high_error_count']}\n")
                f.write(
                    f"Error bias: {insights['error_bias']:.4f} ({insights['over_predictions']} over, {insights['under_predictions']} under)\n")

                f.write("\nTop features correlated with error:\n")
                for feature, corr in insights['correlated_features'].items():
                    f.write(f"  {feature}: {corr:.4f}\n")

                f.write("\nError by feature range:\n")
                for feature, ranges in insights['error_by_value_range'].items():
                    f.write(f"  {feature}:\n")
                    for bin_name, mean_error in ranges.items():
                        f.write(f"    {bin_name}: {mean_error:.4f}\n")

            f.write("\n\n=== Enhanced Model Performance ===\n")
            for model_name, metrics in model_results.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * (len(model_name) + 1) + "\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")

            if self.feature_importance_shift:
                f.write("\n\n=== Feature Importance Shifts ===\n")
                for feature, changes in sorted(self.feature_importance_shift.items(),
                                               key=lambda x: abs(x[1]['importance_change_pct']),
                                               reverse=True)[:10]:
                    f.write(f"{feature}:\n")
                    f.write(f"  Importance change: {changes['importance_change_pct']:.2f}%\n")
                    f.write(
                        f"  Rank change: {changes['rank_change']} (from {changes['old_rank']} to {changes['new_rank']})\n")

            f.write(f"\n\nEnhancement process completed in {time.time() - start_time:.2f} seconds\n")

        print(f"Enhancement report saved to {report_file}")