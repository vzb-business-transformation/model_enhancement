import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import time
import os


class MLModels:
    """
    Class for training and evaluating various machine learning models for regression.
    """

    def __init__(self, random_state=42):
        """
        Initialize ML models.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.best_models = {}
        self.feature_importances = {}
        self.model_params = {}

    def select_features(self, X, y, threshold=0.001, method='rf'):
        """
        Select important features.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            threshold (float): Importance threshold
            method (str): Method for feature selection ('rf', 'xgb', or 'lasso')

        Returns:
            pd.DataFrame: Selected features
        """
        print(f"Selecting features using {method} method...")

        # Check for and handle NaN values
        nan_mask = y.isna()
        if nan_mask.any():
            nan_count = nan_mask.sum()
            print(
                f"Warning: Found {nan_count} NaN values in target variable. Removing these samples for feature selection.")
            valid_indices = ~nan_mask
            X_valid = X.loc[valid_indices]
            y_valid = y.loc[valid_indices]
        else:
            X_valid = X
            y_valid = y

        if method == 'rf':
            selector = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif method == 'xgb':
            selector = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif method == 'lasso':
            selector = Lasso(alpha=0.01, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Fit selector
        selector.fit(X_valid, y_valid)

        # Get feature importances
        if method == 'lasso':
            importance = pd.Series(np.abs(selector.coef_), index=X.columns)
        else:
            importance = pd.Series(selector.feature_importances_, index=X.columns)

        # Store feature importances
        self.feature_importances[method] = importance.sort_values(ascending=False)

        # Select features above threshold
        important_features = importance[importance > threshold].index

        print(f"Selected {len(important_features)} features out of {len(X.columns)}")
        return X[important_features]

    def drop_correlated_features(self, X, threshold=0.95):
        """
        Remove highly correlated features.

        Args:
            X (pd.DataFrame): Feature matrix
            threshold (float): Correlation threshold

        Returns:
            pd.DataFrame: Features with correlations below threshold
        """
        print(f"Removing highly correlated features above {threshold}...")

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Create mask for upper triangle
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with high correlation
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

        print(f"Dropping {len(to_drop)} highly correlated features")

        # Return filtered features
        return X.drop(columns=to_drop)

    def get_model_params(self):
        """
        Define hyperparameter search spaces for different models.

        Returns:
            dict: Hyperparameter search spaces
        """
        rf_params = {
            'n_estimators': [500, 1000, 1500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        gb_params = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        xgb_params = {
            'n_estimators': [200, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }

        lgb_params = {
            'n_estimators': [200, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'min_data_in_leaf': [10, 20, 30]
        }

        cat_params = {
            'iterations': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7]
        }

        linear_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }

        return {
            'rf': rf_params,
            'gb': gb_params,
            'xgb': xgb_params,
            'lgb': lgb_params,
            'cat': cat_params,
            'ridge': linear_params,
            'lasso': linear_params
        }

    def train_model(self, model_type, X_train, y_train, params=None, n_iter=100, cv=5, verbose=1):
        """
        Train a specific model with hyperparameter tuning.

        Args:
            model_type (str): Type of model to train
            X_train: Training features
            y_train: Training targets
            params (dict): Hyperparameters (if None, uses default search space)
            n_iter (int): Number of iterations for RandomizedSearchCV
            cv (int): Number of cross-validation folds
            verbose (int): Verbosity level

        Returns:
            tuple: Best model and best parameters
        """
        print(f"\nTraining {model_type} model...")

        # Initialize model
        if model_type == 'rf':
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        elif model_type == 'gb':
            model = GradientBoostingRegressor(random_state=self.random_state)
        elif model_type == 'xgb':
            model = XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        elif model_type == 'lgb':
            model = LGBMRegressor(random_state=self.random_state, verbose=-1, force_col_wise=True)
        elif model_type == 'cat':
            model = CatBoostRegressor(verbose=0, random_state=self.random_state)
        elif model_type == 'ridge':
            model = Ridge(random_state=self.random_state)
        elif model_type == 'lasso':
            model = Lasso(random_state=self.random_state)
        elif model_type == 'elastic':
            model = ElasticNet(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Use default params if none provided
        if params is None:
            params = self.get_model_params().get(model_type, {})

        # Hyperparameter tuning
        start_time = time.time()
        search = RandomizedSearchCV(
            model,
            params,
            n_iter=n_iter,
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            scoring='r2',
            verbose=verbose
        )

        search.fit(X_train, y_train)

        # Store results
        best_model = search.best_estimator_
        best_params = search.best_params_

        # Save feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importances[model_type] = pd.Series(
                best_model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)

        # Store model parameters
        self.model_params[model_type] = best_params

        # Print training info
        print(f"{model_type} training completed in {time.time() - start_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")

        return best_model, best_params

    def train_all_models(self, X_train, y_train, models_to_train=None):
        """
        Train multiple models.

        Args:
            X_train: Training features
            y_train: Training targets
            models_to_train (list): List of model types to train (if None, trains all)

        Returns:
            dict: Dictionary of trained models
        """
        print("Training multiple models...")

        if models_to_train is None:
            models_to_train = ['rf', 'gb', 'xgb', 'lgb', 'cat']

        trained_models = {}

        for model_type in models_to_train:
            model, _ = self.train_model(model_type, X_train, y_train)
            trained_models[model_type] = model
            self.best_models[model_type] = model

        return trained_models

    def create_stacked_ensemble(self, X_train, X_test, y_train, y_test, base_models=None):
        """
        Create a stacked ensemble model.

        Args:
            X_train, X_test, y_train, y_test: Training and test data
            base_models (dict): Dictionary of base models (if None, uses best models)

        Returns:
            tuple: Trained ensemble model and performance metrics
        """
        print("\nBuilding stacked ensemble model...")

        # Use existing best models if no base models provided
        if base_models is None:
            if not self.best_models:
                raise ValueError("No trained models available. Train models first.")
            base_models = self.best_models

        # Convert to list of tuples for StackingRegressor
        estimators = [(name, model) for name, model in base_models.items()]

        # Create meta-learner
        meta_learner = Ridge(alpha=1.0)

        # Create stacking ensemble
        stacked_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )

        # Train the ensemble
        print("Training stacked ensemble model...")
        stacked_model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = stacked_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)

        print("\nStacked Ensemble Performance:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        # Store model
        self.best_models['stacked_ensemble'] = stacked_model

        return stacked_model, metrics

    def create_voting_ensemble(self, X_train, X_test, y_train, y_test, base_models=None):
        """
        Create a weighted voting ensemble.

        Args:
            X_train, X_test, y_train, y_test: Training and test data
            base_models (dict): Dictionary of base models (if None, uses best models)

        Returns:
            tuple: Trained voting ensemble model and performance metrics
        """
        print("\nCreating weighted voting ensemble...")

        # Use existing best models if no base models provided
        if base_models is None:
            if not self.best_models:
                raise ValueError("No trained models available. Train models first.")
            base_models = self.best_models

        # Convert to list of tuples for VotingRegressor
        estimators = [(name, model) for name, model in base_models.items()]

        # Evaluate individual models to determine weights
        weights = []
        for name, model in base_models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            weights.append(max(0, r2))  # Ensure non-negative weights

        # Normalize weights to sum to 1
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]
        else:
            # Equal weights if all models perform poorly
            weights = [1 / len(base_models) for _ in base_models]

        print(f"Optimized weights: {[round(w, 3) for w in weights]}")

        # Create voting ensemble
        voting_model = VotingRegressor(
            estimators=estimators,
            weights=weights
        )

        # Train the ensemble
        voting_model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = voting_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)

        print("\nWeighted Voting Ensemble Performance:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        # Store model
        self.best_models['voting_ensemble'] = voting_model

        return voting_model, metrics

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted target values

        Returns:
            dict: Dictionary of metrics
        """
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred, squared=False),
            'R2': r2_score(y_true, y_pred)
        }

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            dict: Dictionary of model metrics
        """
        print("\nEvaluating all models...")

        if not self.best_models:
            raise ValueError("No trained models available. Train models first.")

        results = {}

        for name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred)
            results[name] = metrics

            print(f"\n{name} model performance:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        return results

    def plot_feature_importance(self, model_type=None, top_n=20):
        """
        Plot feature importance for selected models.

        Args:
            model_type (str): Model type to plot (if None, plots all available)
            top_n (int): Number of top features to show
        """
        if not self.feature_importances:
            raise ValueError("No feature importance data available. Train models first.")

        # If model_type specified, plot just that model
        if model_type and model_type in self.feature_importances:
            plt.figure(figsize=(12, 6))
            self.feature_importances[model_type].head(top_n).plot(kind='barh')
            plt.title(f'{model_type.upper()} Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
            return

        # Otherwise plot all available models
        num_models = len(self.feature_importances)
        fig, axes = plt.subplots(num_models, 1, figsize=(12, 6 * num_models), constrained_layout=True)

        # Handle case with just one model
        if num_models == 1:
            axes = [axes]

        # Plot each model's feature importance
        for ax, (model_name, importance) in zip(axes, self.feature_importances.items()):
            importance.head(top_n).plot(kind='barh', ax=ax)
            ax.set_title(f'{model_name.upper()} Feature Importance')
            ax.set_xlabel('Importance Score')

        plt.show()

    def save_models(self, directory="results/models"):
        """
        Save all trained models.

        Args:
            directory (str): Directory to save models
        """
        print("\nSaving trained models...")

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save each model
        for name, model in self.best_models.items():
            filepath = os.path.join(directory, f"{name}_model.joblib")
            joblib.dump(model, filepath)
            print(f"Model '{name}' saved to {filepath}")

        # Save feature importances
        if self.feature_importances:
            importance_path = os.path.join(directory, "feature_importances.joblib")
            joblib.dump(self.feature_importances, importance_path)
            print(f"Feature importances saved to {importance_path}")

        # Save model parameters
        if self.model_params:
            params_path = os.path.join(directory, "model_params.joblib")
            joblib.dump(self.model_params, params_path)
            print(f"Model parameters saved to {params_path}")

    def load_models(self, directory="results/models"):
        """
        Load saved models.

        Args:
            directory (str): Directory containing saved models
        """
        print("\nLoading saved models...")

        # Check if directory exists
        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")

        # Load models
        self.best_models = {}
        model_files = [f for f in os.listdir(directory) if f.endswith("_model.joblib")]

        for model_file in model_files:
            model_name = model_file.replace("_model.joblib", "")
            filepath = os.path.join(directory, model_file)
            self.best_models[model_name] = joblib.load(filepath)
            print(f"Loaded model '{model_name}' from {filepath}")

        # Load feature importances
        importance_path = os.path.join(directory, "feature_importances.joblib")
        if os.path.exists(importance_path):
            self.feature_importances = joblib.load(importance_path)
            print(f"Loaded feature importances from {importance_path}")

        # Load model parameters
        params_path = os.path.join(directory, "model_params.joblib")
        if os.path.exists(params_path):
            self.model_params = joblib.load(params_path)
            print(f"Loaded model parameters from {params_path}")

        return self.best_models