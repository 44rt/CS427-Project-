# utils/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from config import DATA_PATH, RANDOM_STATE, TEST_SIZE, VAL_SIZE, TASK_ATTACK_ORDER, BATCH_SIZE

def load_and_preprocess_data():
    """Load and preprocess the encoded data with cybersecurity-aware missing value handling"""
    print("📊 Loading and preprocessing data...")
    df = pd.read_csv(DATA_PATH)
    
    # Create binary label
    df['binary_label'] = (df['Label'] != 'Benign').astype(int)
    
    print("🔍 Analyzing missing values from cybersecurity perspective...")
    print(f"   Initial dataset shape: {df.shape}")
    print(f"   Total missing values: {df.isnull().sum().sum():,}")
    
    # 1. Remove columns that are mostly missing (>80% empty) - these are likely useless
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop = missing_percentage[missing_percentage > 0.8].index
    df = df.drop(columns=columns_to_drop)
    print(f"   Dropped {len(columns_to_drop)} columns with >80% missing data")
    
    # 2. Remove rows that are completely empty (no information value)
    initial_rows = len(df)
    df = df.dropna(how='all')
    rows_dropped = initial_rows - len(df)
    print(f"   Dropped {rows_dropped} completely empty rows")
    
    # 3. Intelligent cybersecurity-aware imputation for remaining missing values
    print("   Applying cybersecurity-aware imputation...")
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # For NUMERIC features (packet sizes, durations, timings):
    # Use median (robust against attack outliers)
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        print(f"   Applied median imputation to {len(numeric_cols)} numeric features")
    
    # For CATEGORICAL features (protocols, flags, states):
    # Use 'unknown' category - this is the cybersecurity insight!
    # Missing values might indicate anomalous behavior
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        print(f"   Created 'unknown' category for {len(categorical_cols)} categorical features")
    
    # 4. Final validation
    final_missing = df.isnull().sum().sum()
    print(f"✅ Final dataset shape: {df.shape}")
    print(f"✅ Remaining missing values: {final_missing}")
    
    if final_missing > 0:
        print("⚠️  Warning: Still some missing values. This might indicate mixed data types in columns.")
        # As a fallback, drop any remaining missing values
        df = df.dropna()
        print(f"   Dropped remaining {final_missing} missing values")
        print(f"   Final cleaned shape: {df.shape}")
    
    return df


def create_incremental_tasks():
    """Create incremental learning tasks with proper test splits"""
    df = load_and_preprocess_data()
    
    # Create global test set with all attack types
    main_df, global_test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=df['Attack Type']
    )
    
    # Prepare features and labels for global test
    X_global_test = global_test_df.drop(['Label', 'Attack Type', 'Attack Tool', 'binary_label'], axis=1, errors='ignore')
    y_global_test = global_test_df['binary_label']
    
    # Create task-specific datasets
    tasks = []
    task_test_sets = []
    
    for i, attack_list in enumerate(TASK_ATTACK_ORDER):
        # Get task data
        task_data = main_df[main_df['Attack Type'].isin(['Benign'] + attack_list)]
        
        # Split into train/validation
        X_task = task_data.drop(['Label', 'Attack Type', 'Attack Tool', 'binary_label'], axis=1, errors='ignore')
        y_task = task_data['binary_label']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_task, y_task, test_size=VAL_SIZE, random_state=RANDOM_STATE,
            stratify=task_data['Attack Type']
        )
        
        # Create task-specific test set
        _, task_test = train_test_split(
            task_data, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=task_data['Attack Type']
        )
        
        X_task_test = task_test.drop(['Label', 'Attack Type', 'Attack Tool', 'binary_label'], axis=1, errors='ignore')
        y_task_test = task_test['binary_label']
        
        tasks.append((X_train, y_train, X_val, y_val))
        task_test_sets.append((X_task_test, y_task_test))
    
    return tasks, task_test_sets, (X_global_test, y_global_test)

def create_torch_dataloaders(tasks):
    """Convert sklearn datasets to PyTorch DataLoaders"""
    torch_tasks = []
    
    for X_train, y_train, X_val, y_val in tasks:
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        torch_tasks.append((train_loader, val_loader))
    
    return torch_tasks