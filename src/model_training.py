from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from src.logger import logger

def evaluate_model(model, X_test, y_test, model_name):
    logger.info(f"Evaluating {model_name} model...")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"{model_name} Accuracy: {accuracy:.4f}")

    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"], digits=3)
    logger.info(f"Classification Report for {model_name}:\n{report}")

    roc_auc = roc_auc_score(y_test, y_proba)
    logger.info(f"{model_name} ROC-AUC: {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plot_roc_curve(fpr, tpr, model_name)

    return accuracy, roc_auc

def plot_roc_curve(fpr, tpr, model_name):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
