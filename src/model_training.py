from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from src.logger import logger

def train_logistic_regression(X_train, y_train):
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500, class_weight="balanced", solver="saga")
    model.fit(X_train, y_train)
    logger.info("Logistic Regression model trained.")
    return model

def train_svm(X_train, y_train):
    logger.info("Training Support Vector Machine model...")
    model = SVC(probability=True, class_weight="balanced")
    model.fit(X_train, y_train)
    logger.info("Support Vector Machine model trained.")
    return model

def train_random_forest(X_train, y_train):
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    logger.info("Random Forest model trained.")
    return model

def train_xgboost(X_train, y_train):
    logger.info("Training XGBoost model...")
    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    logger.info("XGBoost model trained.")
    return model

def evaluate_model(model, X_test, y_test, model_name):
    logger.info(f"Evaluating {model_name} model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"{model_name} Accuracy: {accuracy:.4f}")
    return accuracy
