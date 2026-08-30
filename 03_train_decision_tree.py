import time
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def train_decision_tree():
    print('loading preprocessed arrays...')
    data = np.load('processed_data.npz')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    w_train = data['w_train']

    print('Training Decision Tree Classifier (max_depth=4)...')
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=35)

    start_time = time.time()
    dt_model.fit(X_train, y_train, sample_weight=w_train)
    elapsed_time = time.time() - start_time

    print(f'Training completed in {elapsed_time:.4f} seconds.')

    # evaluating probabilities on test set
    y_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'Decision Tree ROC-AUC Score: {roc_auc:.4f}')

    # Save trained model artifact
    joblib.dump(dt_model, "dt_model.joblib")
    print("Model saved to 'dt_model.joblib'. ")

if __name__ == "__main__":
    train_decision_tree()

