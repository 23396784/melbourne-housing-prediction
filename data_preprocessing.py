"""
Melbourne Housing Price Prediction - Data Preprocessing Module
================================================================
This module handles data cleaning, feature engineering, and preparation
for machine learning models.

Author: Victor Prefa
Course: SIG720 Machine Learning, Deakin University
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class MelbourneHousingPreprocessor:
    """
    Comprehensive data preprocessing pipeline for Melbourne housing data.
    
    Handles:
    - Data loading and validation
    - Missing value treatment
    - Feature engineering
    - Encoding categorical variables
    - Feature scaling
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Sold_Price'
        
    def load_data(self, filepath):
        """
        Load and perform initial validation of the dataset.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded and validated dataframe
        """
        print("=" * 60)
        print("STEP 1: DATA LOADING")
        print("=" * 60)
        
        df = pd.read_csv(filepath)
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Columns: {list(df.columns)}")
        
        return df
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values, duplicates, and outliers.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataframe
            
        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe
        """
        print("\n" + "=" * 60)
        print("STEP 2: DATA CLEANING")
        print("=" * 60)
        
        initial_rows = len(df)
        
        # Handle date column
        if 'Sold_Date' in df.columns:
            df['Sold_Date'] = pd.to_datetime(df['Sold_Date'], errors='coerce')
            df = df.dropna(subset=['Sold_Date'])
            
        # Remove rows with missing target
        df = df.dropna(subset=[self.target_column])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Filter reasonable price range (remove extreme outliers)
        q1 = df[self.target_column].quantile(0.01)
        q99 = df[self.target_column].quantile(0.99)
        df = df[(df[self.target_column] >= q1) & (df[self.target_column] <= q99)]
        
        final_rows = len(df)
        
        print(f"✅ Data cleaning complete")
        print(f"   Rows removed: {initial_rows - final_rows}")
        print(f"   Final dataset size: {final_rows} rows")
        
        return df
    
    def engineer_features(self, df):
        """
        Create new features from existing data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with engineered features
        """
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 60)
        
        # Temporal features from Sold_Date
        if 'Sold_Date' in df.columns:
            df['Sold_Year'] = df['Sold_Date'].dt.year
            df['Sold_Month'] = df['Sold_Date'].dt.month
            df['Sold_Quarter'] = df['Sold_Date'].dt.quarter
            df['Years_Since_2000'] = df['Sold_Year'] - 2000
            df['Is_Recent'] = (df['Sold_Year'] >= 2020).astype(int)
            print("   ✓ Created temporal features: Year, Month, Quarter")
        
        # School access level based on suburb
        school_counts = {
            'Richmond': 8,
            'South Yarra': 6,
            'Hawthorn': 10
        }
        
        if 'Suburb' in df.columns:
            df['Schools_Nearby'] = df['Suburb'].map(school_counts).fillna(5)
            df['School_Access_Level'] = pd.cut(
                df['Schools_Nearby'], 
                bins=[0, 5, 8, 15], 
                labels=['Low', 'Medium', 'High']
            )
            print("   ✓ Created school access features")
        
        # Property size indicator
        if 'Bedrooms' in df.columns and 'Bathrooms' in df.columns:
            df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
            df['Room_Ratio'] = df['Bedrooms'] / (df['Bathrooms'] + 1)
            print("   ✓ Created room-based features")
        
        # Distance categories
        if 'Distance_to_CBD' in df.columns:
            df['Distance_Category'] = pd.cut(
                df['Distance_to_CBD'],
                bins=[0, 5, 10, 20, 100],
                labels=['Very Close', 'Close', 'Moderate', 'Far']
            )
            print("   ✓ Created distance category feature")
        
        print(f"\n✅ Feature engineering complete")
        print(f"   Total features: {df.shape[1]}")
        
        return df
    
    def encode_categorical(self, df):
        """
        Encode categorical variables using appropriate methods.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with categorical columns
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded categorical variables
        """
        print("\n" + "=" * 60)
        print("STEP 4: CATEGORICAL ENCODING")
        print("=" * 60)
        
        categorical_cols = ['Suburb', 'Property_Type']
        
        for col in categorical_cols:
            if col in df.columns:
                # One-hot encoding for low cardinality columns
                if df[col].nunique() <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                    df = pd.concat([df, dummies], axis=1)
                    print(f"   ✓ One-hot encoded: {col} ({df[col].nunique()} categories)")
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"   ✓ Label encoded: {col}")
        
        print(f"\n✅ Categorical encoding complete")
        
        return df
    
    def prepare_features(self, df, target_col='Sold_Price'):
        """
        Prepare final feature matrix and target vector.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Fully processed dataframe
        target_col : str
            Name of target column
            
        Returns:
        --------
        tuple
            (X, y, feature_names)
        """
        print("\n" + "=" * 60)
        print("STEP 5: FEATURE PREPARATION")
        print("=" * 60)
        
        # Select numeric columns for modeling
        exclude_cols = [
            target_col, 'Property_ID', 'Address', 'Agency', 
            'Sold_Date', 'Suburb', 'Property_Type', 
            'School_Access_Level', 'Distance_Category'
        ]
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['int64', 'float64', 'bool', 'uint8']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Remove constant features
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            print(f"   ⚠ Removed constant features: {constant_cols}")
        
        self.feature_columns = list(X.columns)
        
        print(f"✅ Feature preparation complete")
        print(f"   Features selected: {len(self.feature_columns)}")
        print(f"   Feature names: {self.feature_columns}")
        print(f"   Target shape: {y.shape}")
        
        return X, y, self.feature_columns
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame, optional
            Test features
            
        Returns:
        --------
        tuple
            Scaled (X_train, X_test) or just X_train if X_test is None
        """
        print("\n" + "=" * 60)
        print("STEP 6: FEATURE SCALING")
        print("=" * 60)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        print(f"✅ Training features scaled")
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            print(f"✅ Test features scaled")
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_summary_statistics(self, df):
        """
        Generate summary statistics for the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to summarize
            
        Returns:
        --------
        dict
            Summary statistics
        """
        summary = {
            'total_properties': len(df),
            'suburbs': df['Suburb'].unique().tolist() if 'Suburb' in df.columns else [],
            'price_range': {
                'min': df[self.target_column].min(),
                'max': df[self.target_column].max(),
                'mean': df[self.target_column].mean(),
                'median': df[self.target_column].median()
            }
        }
        
        if 'Suburb' in df.columns:
            summary['suburb_stats'] = df.groupby('Suburb')[self.target_column].agg([
                'count', 'mean', 'median', 'std'
            ]).to_dict()
        
        return summary


def main():
    """Main execution function for preprocessing pipeline."""
    
    # Initialize preprocessor
    preprocessor = MelbourneHousingPreprocessor()
    
    # Load data
    df = preprocessor.load_data('../data/melbourne_housing.csv')
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Engineer features
    df = preprocessor.engineer_features(df)
    
    # Encode categorical variables
    df = preprocessor.encode_categorical(df)
    
    # Prepare features
    X, y, feature_names = preprocessor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


if __name__ == "__main__":
    main()
