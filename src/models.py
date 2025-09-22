"""
Machine Learning Models for Gas Monitoring Data Analysis
AIAP 21 Technical Assessment - Part 2

This module implements three machine learning models for gas monitoring data:
1. Logistic Regression
2. Random Forest
3. XGBoost (Gradient Boosting)

Target Variables: Activity Level or HVAC Operation Mode
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import KNNImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class GasMonitoringMLPipeline:
    """
    Complete machine learning pipeline for gas monitoring data analysis.
    """
    
    def __init__(self, target_column='Activity Level'):
        """
        Initialize the ML pipeline.
        
        Args:
            target_column (str): Target variable for prediction
        """
        self.target_column = target_column
        self.scaler = RobustScaler()  # Robust to outliers
        self.label_encoder = LabelEncoder()
        self.imputer = KNNImputer(n_neighbors=5)
        self.models = {}
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self, df):
        """
        Load and preprocess the gas monitoring data.
        
        Args:
            df (pd.DataFrame): Raw gas monitoring data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        print("Loading and preprocessing data...")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # 1. Handle missing values using KNN imputation
        print("   - Handling missing values with KNN imputation...")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = self.imputer.fit_transform(data[numeric_columns])
        
        # 2. Handle categorical missing values
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != self.target_column:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
        
        # 3. Remove duplicates
        print("   - Removing duplicate rows...")
        data = data.drop_duplicates()
        
        # 4. Handle outliers using IQR method
        print("   - Handling outliers...")
        for col in numeric_columns:
            if col != 'Session ID':  # Don't cap Session ID
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 5. Feature Engineering
        print("   - Creating engineered features...")
        data = self._create_features(data)
        
        # 6. Encode categorical variables
        print("   - Encoding categorical variables...")
        data = self._encode_categorical(data)
        
        print(f"Data preprocessing completed. Shape: {data.shape}")
        return data
    
    def _create_features(self, data):
        """
        Create engineered features for better model performance.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        # 1. Sensor interaction features
        if 'CO2_InfraredSensor' in data.columns and 'CO2_ElectroChemicalSensor' in data.columns:
            data['CO2_Sensor_Diff'] = data['CO2_ElectroChemicalSensor'] - data['CO2_InfraredSensor']
            data['CO2_Sensor_Ratio'] = data['CO2_ElectroChemicalSensor'] / (data['CO2_InfraredSensor'] + 1e-8)
        
        # 2. Metal oxide sensor statistics
        metal_oxide_cols = [col for col in data.columns if 'MetalOxideSensor' in col]
        if metal_oxide_cols:
            data['MetalOxide_Mean'] = data[metal_oxide_cols].mean(axis=1)
            data['MetalOxide_Std'] = data[metal_oxide_cols].std(axis=1)
            data['MetalOxide_Max'] = data[metal_oxide_cols].max(axis=1)
            data['MetalOxide_Min'] = data[metal_oxide_cols].min(axis=1)
        
        # 3. Environmental comfort index
        if 'Temperature' in data.columns and 'Humidity' in data.columns:
            # Temperature comfort zone: 20-25°C
            data['Temp_Comfort'] = np.where(
                (data['Temperature'] >= 20) & (data['Temperature'] <= 25), 1, 0
            )
            # Humidity comfort zone: 40-60%
            data['Humidity_Comfort'] = np.where(
                (data['Humidity'] >= 40) & (data['Humidity'] <= 60), 1, 0
            )
            data['Comfort_Index'] = data['Temp_Comfort'] + data['Humidity_Comfort']
        
        # 4. Time-based features (if Time of Day exists)
        if 'Time of Day' in data.columns:
            time_mapping = {'morning': 1, 'afternoon': 2, 'evening': 3, 'night': 4}
            data['Time_Numeric'] = data['Time of Day'].map(time_mapping)
        
        return data
    
    def _encode_categorical(self, data):
        """
        Encode categorical variables for machine learning.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
        """
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != self.target_column:
                # One-hot encoding for categorical variables
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data, dummies], axis=1)
                data = data.drop(col, axis=1)
        
        return data
    
    def prepare_features_and_target(self, data):
        """
        Prepare features and target variable for training.
        
        Args:
            data (pd.DataFrame): Preprocessed data
        """
        print("Preparing features and target variable...")
        
        # Define feature columns (exclude target and non-predictive columns)
        exclude_columns = [self.target_column, 'Session ID']
        self.feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Prepare features and target
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   - Features: {len(self.feature_columns)} columns")
        print(f"   - Training set: {self.X_train.shape[0]} samples")
        print(f"   - Test set: {self.X_test.shape[0]} samples")
        print(f"   - Target classes: {len(np.unique(y_encoded))}")
    
    def train_logistic_regression(self):
        """
        Train Logistic Regression model.
        
        Returns:
            dict: Model performance metrics
        """
        print("\nTraining Logistic Regression...")
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Best model
        best_lr = grid_search.best_estimator_
        self.models['Logistic Regression'] = best_lr
        
        # Predictions
        y_pred = best_lr.predict(self.X_test_scaled)
        
        # Performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        cv_scores = cross_val_score(best_lr, self.X_train_scaled, self.y_train, cv=5)
        
        performance = {
            'model': 'Logistic Regression',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(self.y_test, y_pred, 
                                                        target_names=self.label_encoder.classes_)
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return performance
    
    def train_random_forest(self):
        """
        Train Random Forest model.
        
        Returns:
            dict: Model performance metrics
        """
        print("\nTraining Random Forest...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        self.models['Random Forest'] = best_rf
        
        # Predictions
        y_pred = best_rf.predict(self.X_test)
        
        # Performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        cv_scores = cross_val_score(best_rf, self.X_train, self.y_train, cv=5)
        
        performance = {
            'model': 'Random Forest',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'feature_importance': dict(zip(self.feature_columns, best_rf.feature_importances_)),
            'classification_report': classification_report(self.y_test, y_pred,
                                                        target_names=self.label_encoder.classes_)
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return performance
    
    def train_xgboost(self):
        """
        Train XGBoost model.
        
        Returns:
            dict: Model performance metrics
        """
        print("\nTraining XGBoost...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        best_xgb = grid_search.best_estimator_
        self.models['XGBoost'] = best_xgb
        
        # Predictions
        y_pred = best_xgb.predict(self.X_test)
        
        # Performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        cv_scores = cross_val_score(best_xgb, self.X_train, self.y_train, cv=5)
        
        performance = {
            'model': 'XGBoost',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'feature_importance': dict(zip(self.feature_columns, best_xgb.feature_importances_)),
            'classification_report': classification_report(self.y_test, y_pred,
                                                        target_names=self.label_encoder.classes_)
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return performance
    
    def train_all_models(self):
        """
        Train all three models and return performance comparison.
        
        Returns:
            dict: Performance comparison of all models
        """
        print("Training all models...")
        print("=" * 50)
        
        performances = []
        
        # Train each model
        performances.append(self.train_logistic_regression())
        performances.append(self.train_random_forest())
        performances.append(self.train_xgboost())
        
        # Create comparison
        comparison = pd.DataFrame([
            {
                'Model': perf['model'],
                'Test Accuracy': perf['accuracy'],
                'CV Mean': perf['cv_mean'],
                'CV Std': perf['cv_std']
            }
            for perf in performances
        ])
        
        print("\nMODEL PERFORMANCE COMPARISON")
        print("=" * 50)
        print(comparison.to_string(index=False))
        
        # Find best model
        best_model = comparison.loc[comparison['Test Accuracy'].idxmax()]
        print(f"\nBest Model: {best_model['Model']} (Accuracy: {best_model['Test Accuracy']:.4f})")
        
        return {
            'performances': performances,
            'comparison': comparison,
            'best_model': best_model
        }
    
    def get_model_explanations(self):
        """
        Get explanations for why these three models were chosen.
        
        Returns:
            dict: Model selection explanations
        """
        explanations = {
            'Logistic Regression': {
                'why_chosen': [
                    'Linear baseline model for classification',
                    'Fast training and prediction',
                    'Provides probability estimates',
                    'Good for understanding feature relationships',
                    'Works well with scaled features'
                ],
                'pros': [
                    'Interpretable coefficients',
                    'Fast and efficient',
                    'No hyperparameter tuning complexity',
                    'Good baseline performance'
                ],
                'cons': [
                    'Assumes linear relationship',
                    'May not capture complex patterns',
                    'Sensitive to outliers'
                ]
            },
            'Random Forest': {
                'why_chosen': [
                    'Handles non-linear relationships well',
                    'Robust to outliers and missing values',
                    'Provides feature importance',
                    'Good for mixed data types',
                    'Reduces overfitting through ensemble'
                ],
                'pros': [
                    'High accuracy on many problems',
                    'Feature importance ranking',
                    'Handles missing values naturally',
                    'Less prone to overfitting'
                ],
                'cons': [
                    'Less interpretable than linear models',
                    'Can be memory intensive',
                    'May overfit with many trees'
                ]
            },
            'XGBoost': {
                'why_chosen': [
                    'State-of-the-art gradient boosting',
                    'Excellent performance on tabular data',
                    'Built-in regularization',
                    'Handles missing values well',
                    'Often wins ML competitions'
                ],
                'pros': [
                    'Very high accuracy',
                    'Fast training and prediction',
                    'Built-in feature importance',
                    'Handles missing values automatically'
                ],
                'cons': [
                    'Many hyperparameters to tune',
                    'Can overfit with small datasets',
                    'Less interpretable'
                ]
            }
        }
        
        return explanations

def main():
    """
    Main function to demonstrate the ML pipeline.
    """
    print("Gas Monitoring Machine Learning Pipeline")
    print("=" * 60)
    
    # This would be called from the main analysis script
    print("This module is designed to be imported and used in the main analysis.")
    print("Example usage:")
    print("  from src.models import GasMonitoringMLPipeline")
    print("  pipeline = GasMonitoringMLPipeline(target_column='Activity Level')")
    print("  data = pipeline.load_and_preprocess_data(df)")
    print("  pipeline.prepare_features_and_target(data)")
    print("  results = pipeline.train_all_models()")

if __name__ == "__main__":
    main()
