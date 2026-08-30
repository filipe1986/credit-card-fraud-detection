import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

def evaluate_models():
    print('Loading preprocessed dataset and trained models...')
    data = np.load('processed_data.npz')
    X_test, y_test = data['X_test'], data['y_test']

    dt_model = joblib.load('dt_model.joblib')
    svm_model = joblib.load('svm_model.joblib')

    print('\n--- Decision Tree Classification Report ---')
    y_pred_dt = dt_model.predict(X_test)
    print(classification_report(y_test, y_pred_dt, target_names=['Legitimate', 'Fraud']))

    print('\n--- Support Vector Machine Classification Report ---')
    y_pred_svm = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred_svm, target_names=['Legitimate', 'Fraud']))

    # Generate and save Confusion Matrix Plot for SVM
    print('Generating Confusion Matrix visualization...')
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_svm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('SVM Fraud Detection - Confusion Matrix')
    plt.tight_layout()

    # Saving the visualization to disk for GitHub README & Linkedin
    plt.savefig('confusion_matrix.png', dpi=300)
    print('Saved plot to "confusion_matrix.png". Evaluation complete!')

if __name__ == "__main__":
    evaluate_models()

