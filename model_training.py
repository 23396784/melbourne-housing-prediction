"""
Melbourne Housing Price Prediction - Model Training Module
============================================================
This module implements multiple regression models, cross-validation,
hyperparameter tuning, and model evaluation.

Author: Victor Prefa
Course: SIG720 Machine Learning, Deakin University
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP not available. Install with: pip install shap")


class MelbourneHousingModels:
    """
    Machine learning model training and evaluation for Melbourne housing prices.
    
    Models implemented:
    - Linear Regression (baseline)
    - Ridge Regression (L2 regularization)
    - Decision Tree Regressor
    - Random Forest Regressor
    - XGBoost Regressor
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
    def initialize_models(self):
        """Initialize all regression models with default parameters."""
        
        print("=" * 60)
        print("INITIALIZING MODELS")
        print("=" * 60)
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=10, 
                min_samples_split=5,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=0
            )
        
        print(f"✅ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"   • {name}")
        
        return self.models
    
    def cross_validate_models(self, X, y, cv=5):
        """
        Perform k-fold cross-validation on all models.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target vector
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        pd.DataFrame
            Cross-validation results for all models
        """
        print("\n" + "=" * 60)
        print(f"{cv}-FOLD CROSS-VALIDATION")
        print("=" * 60)
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        results_list = []
        
        for name, model in self.models.items():
            print(f"\n📊 Evaluating: {name}")
            
            # Cross-validation scores
            mae_scores = []
            rmse_scores = []
            r2_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
                X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold)
                
                # Predict
                y_pred = model.predict(X_val_fold)
                
                # Calculate metrics
                mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
                r2_scores.append(r2_score(y_val_fold, y_pred))
            
            # Store results
            result = {
                'Model': name,
                'MAE': np.mean(mae_scores),
                'MAE_Std': np.std(mae_scores),
                'RMSE': np.mean(rmse_scores),
                'RMSE_Std': np.std(rmse_scores),
                'R²': np.mean(r2_scores),
                'R²_Std': np.std(r2_scores)
            }
            results_list.append(result)
            
            print(f"   MAE:  ${result['MAE']:,.2f} (±${result['MAE_Std']:,.2f})")
            print(f"   RMSE: ${result['RMSE']:,.2f} (±${result['RMSE_Std']:,.2f})")
            print(f"   R²:   {result['R²']:.4f} (±{result['R²_Std']:.4f})")
        
        self.results = pd.DataFrame(results_list)
        
        # Identify best model
        best_idx = self.results['R²'].idxmax()
        self.best_model_name = self.results.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "=" * 60)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"   R² Score: {self.results.loc[best_idx, 'R²']:.4f}")
        print("=" * 60)
        
        return self.results
    
    def hyperparameter_tuning(self, X, y, model_name='Random Forest'):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        model_name : str
            Name of model to tune
            
        Returns:
        --------
        dict
            Best parameters and score
        """
        print("\n" + "=" * 60)
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print("=" * 60)
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'Decision Tree': {
                'max_depth': [3, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        if model_name not in param_grids:
            print(f"⚠ No parameter grid defined for {model_name}")
            return None
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        print(f"   Searching parameter space...")
        print(f"   Parameters: {list(param_grid.keys())}")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"\n✅ Best Parameters: {grid_search.best_params_}")
        print(f"   Best R² Score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[f'{model_name} (Tuned)'] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def train_final_model(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate the final model on test data.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
            
        Returns:
        --------
        dict
            Final evaluation metrics
        """
        print("\n" + "=" * 60)
        print("FINAL MODEL TRAINING & EVALUATION")
        print("=" * 60)
        
        # Train best model on full training set
        self.best_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.best_model.predict(X_train)
        y_test_pred = self.best_model.predict(X_test)
        
        # Training metrics
        train_metrics = {
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'R²': r2_score(y_train, y_train_pred)
        }
        
        # Test metrics
        test_metrics = {
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'R²': r2_score(y_test, y_test_pred)
        }
        
        print(f"\n📈 Training Performance ({self.best_model_name}):")
        print(f"   MAE:  ${train_metrics['MAE']:,.2f}")
        print(f"   RMSE: ${train_metrics['RMSE']:,.2f}")
        print(f"   R²:   {train_metrics['R²']:.4f}")
        
        print(f"\n📊 Test Performance ({self.best_model_name}):")
        print(f"   MAE:  ${test_metrics['MAE']:,.2f}")
        print(f"   RMSE: ${test_metrics['RMSE']:,.2f}")
        print(f"   R²:   {test_metrics['R²']:.4f}")
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': y_test_pred
        }
    
    def get_feature_importance(self, X, feature_names=None):
        """
        Extract feature importance from tree-based models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        pd.DataFrame
            Feature importance rankings
        """
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print(f"\n🔍 Top Features ({self.best_model_name}):")
            for idx, row in self.feature_importance.head(10).iterrows():
                bar = '█' * int(row['Importance'] * 50)
                print(f"   {row['Feature']:<25} {row['Importance']:.4f} {bar}")
        else:
            print(f"⚠ {self.best_model_name} does not support feature importance")
            
            # Use coefficient magnitude for linear models
            if hasattr(self.best_model, 'coef_'):
                importance = np.abs(self.best_model.coef_)
                self.feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance / importance.sum()
                }).sort_values('Importance', ascending=False)
                
                print(f"\n🔍 Coefficient Importance (normalized):")
                for idx, row in self.feature_importance.head(10).iterrows():
                    print(f"   {row['Feature']:<25} {row['Importance']:.4f}")
        
        return self.feature_importance
    
    def shap_analysis(self, X, sample_size=100):
        """
        Perform SHAP analysis for model interpretability.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        sample_size : int
            Number of samples for SHAP analysis
            
        Returns:
        --------
        shap.Explanation or None
        """
        if not SHAP_AVAILABLE:
            print("⚠ SHAP not available. Skipping SHAP analysis.")
            return None
        
        print("\n" + "=" * 60)
        print("SHAP ANALYSIS")
        print("=" * 60)
        
        # Sample data for faster computation
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=self.random_state)
        else:
            X_sample = X
        
        print(f"   Computing SHAP values for {len(X_sample)} samples...")
        
        try:
            explainer = shap.TreeExplainer(self.best_model)
            shap_values = explainer.shap_values(X_sample)
            
            print("✅ SHAP analysis complete")
            
            # Summary of mean absolute SHAP values
            mean_shap = pd.DataFrame({
                'Feature': X_sample.columns,
                'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
            }).sort_values('Mean |SHAP|', ascending=False)
            
            print("\n🔍 SHAP Feature Importance:")
            for idx, row in mean_shap.head(10).iterrows():
                print(f"   {row['Feature']:<25} {row['Mean |SHAP|']:.4f}")
            
            return shap_values
            
        except Exception as e:
            print(f"⚠ SHAP analysis failed: {e}")
            return None
    
    def predict(self, X):
        """
        Make predictions using the best trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features for prediction
            
        Returns:
        --------
        np.ndarray
            Predicted prices
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run cross_validate_models first.")
        
        return self.best_model.predict(X)
    
    def get_results_summary(self):
        """
        Get a formatted summary of all model results.
        
        Returns:
        --------
        str
            Formatted results summary
        """
        if self.results is None or len(self.results) == 0:
            return "No results available. Run cross_validate_models first."
        
        summary = "\n" + "=" * 60 + "\n"
        summary += "MODEL COMPARISON SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        # Format results table
        summary += self.results.to_string(index=False, float_format=lambda x: f'{x:,.2f}')
        
        summary += f"\n\n🏆 Best Model: {self.best_model_name}\n"
        
        return summary


def main():
    """Main execution function for model training pipeline."""
    
    # Example usage (requires preprocessed data)
    print("Melbourne Housing Price Prediction - Model Training")
    print("=" * 60)
    
    # Initialize model trainer
    trainer = MelbourneHousingModels(random_state=42)
    
    # Initialize models
    trainer.initialize_models()
    
    print("\n✅ Model training module loaded successfully")
    print("   Use trainer.cross_validate_models(X, y) to evaluate models")
    print("   Use trainer.hyperparameter_tuning(X, y) for optimization")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
