import time
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss, roc_auc_score

def train_svm():
    print('Loading preprocessed arrays...')
    data = np.load('processed_data.npz')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    w_train = data['w_train']

    print('Training Support Vector Machine (LinearSVC) ...')
    svm_model = LinearSVC(class_weight='balanced', random_state=31, max_iter=2000, dual='auto')

    start_time = time.time()
    svm_model.fit(X_train, y_train, sample_weight=w_train)
    elapsed_time = time.time() - start_time

    print(f'Traning completed in {elapsed_time:.4f} seconds.')

    # Compute decision function socres and hinge loss
    decision_scores = svm_model.decision_function(X_test)
    loss = hinge_loss(y_test, decision_scores)
    roc_auc = roc_auc_score(y_test, decision_scores)

    print(f'SVM Hinge Loss: {loss:.5f}')
    print(f'SVM ROC-AUC Score: {roc_auc:.4f}')

    # Saving the trained model artifact
    joblib.dump(svm_model, 'svm_model.joblib')
    print('Model saved to "svm_model.joblib".')

if __name__ == '__main__':
    train_svm()
