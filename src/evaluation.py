from sklearn.metrics import classification_report, confusion_matrix
from src.logger import logger

def evaluate_performance(y_test, y_pred, model_name):
    logger.info(f"Evaluating performance of {model_name}...")
    
    report = classification_report(y_test, y_pred, digits=3)
    logger.info(f"Classification Report for {model_name}:\n{report}")
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix for {model_name}:\n{cm}")
