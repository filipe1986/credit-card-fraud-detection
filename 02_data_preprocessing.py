import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.class_weight import compute_sample_weight

def preprocess_data():
    print("Loading raw dataset...")
    raw_data = pd.read_csv("creditcard.csv")
    
    # Standardize 'Amount' feature to match normalized PCA features
    raw_data['Amount'] = StandardScaler().fit_transform(raw_data[['Amount']])
    
    # Feature matrix X (drop Time and Class) and target y
    X = raw_data.drop(columns=['Time', 'Class']).values
    y = raw_data['Class'].values
    
    # L1 normalization on feature matrix (as done in IBM lab)
    X = normalize(X, norm='l1')
    
    print("Splitting train and test sets (70/30 ratio)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Compute sample weights to balance the severe class imbalance during training
    w_train = compute_sample_weight('balanced', y_train)
    
    print("Saving processed arrays...")
    np.savez_compressed(
        "processed_data.npz", 
        X_train=X_train, X_test=X_test, 
        y_train=y_train, y_test=y_test, 
        w_train=w_train
    )
    print("Preprocessing complete! Data saved to 'processed_data.npz'.")

if __name__ == "__main__":
    preprocess_data()
    