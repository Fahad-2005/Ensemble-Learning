
import numpy as np
import pandas as pd

ROLL_SEED = 47  
np.random.seed(ROLL_SEED)

def add_noise_features(X_inf, total_d):
    n, current_d = X_inf.shape
    if total_d <= current_d:
        return X_inf
    X_noise = np.random.normal(0.0, 1.5, size=(n, total_d - current_d))
    return np.hstack((X_inf, X_noise))

def generate_bagging_datasets():
    """Generates a high-variance dataset with 10% label noise (n=2000, d=15)"""
    n, d = 2000, 15
    X_inf = np.random.normal(0, 1, (n, 6))
    y = (X_inf[:, 0] * X_inf[:, 1] > 0).astype(int)
    
    # Introduce 10% label noise to create unstable decision boundaries
    flip_idx = np.random.choice(n, size=int(0.10 * n), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    
    X = add_noise_features(X_inf, d)
    return X, y

def generate_feature_dominance_dataset():
    """Generates a dataset where one feature completely dominates the target signal"""
    n, d = 1500, 12
    X_inf = np.random.normal(0, 1, (n, 5))
    # Feature 0 is overwhelmingly predictive
    y = (X_inf[:, 0] > 0.3).astype(int)
    X = add_noise_features(X_inf, d)
    return X, y

def generate_class_imbalance_dataset():
    """Generates an imbalanced dataset where majority class >= 4x minority class"""
    n, d = 5500, 15 # Satisfies the n >= 5000 constraint easily
    X_inf = np.random.normal(0, 1, (n, 7))
    # Make a tight decision boundary to restrict positive class instances
    y = (X_inf[:, 0] + X_inf[:, 1] > 1.8).astype(int) 
    
    X = add_noise_features(X_inf, d)
    # Validate imbalance ratio dynamically
    majority_count = np.sum(y == 0)
    minority_count = np.sum(y == 1)
    if majority_count < 4 * minority_count:
        # Fallback to force imbalance if random distribution is too generous
        y = np.zeros(n, dtype=int)
        y[np.random.choice(n, size=1000, replace=False)] = 1
        
    return X, y

def generate_extreme_high_dimensional_dataset():
    """Generates a dataset containing exactly 1000 purely noisy features"""
    n = 1000
    X_inf = np.random.normal(0, 1, (n, 10))
    y = (np.sum(X_inf[:, :5], axis=1) > 0).astype(int)
    X_noise = np.random.normal(0, 1, (n, 1000)) # 1000 noisy columns
    X = np.hstack((X_inf, X_noise))
    return X, y

def generate_adaboost_noise_datasets():
    """Generates base clean distributions alongside explicit noise vectors for AdaBoost"""
    n, d = 1200, 12
    X_inf = np.random.normal(0, 1, (n, 6))
    y_clean = (X_inf[:, 0] + X_inf[:, 1] - X_inf[:, 2] > 0.0).astype(int)
    X = add_noise_features(X_inf, d)
    
    # Generate 5%, 15%, and 30% label variations safely
    y_5 = y_clean.copy()
    y_5[np.random.choice(n, int(0.05*n), replace=False)] ^= 1
    
    y_15 = y_clean.copy()
    y_15[np.random.choice(n, int(0.15*n), replace=False)] ^= 1
    
    y_30 = y_clean.copy()
    y_30[np.random.choice(n, int(0.30*n), replace=False)] ^= 1
    
    return X, y_5, y_15, y_30