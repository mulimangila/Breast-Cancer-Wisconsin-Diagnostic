from src.logger import logger
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_logistic_regression, train_svm, train_random_forest, train_xgboost, evaluate_model
from src.evaluation import evaluate_performance

def main():
    logger.info("Starting the Breast Cancer Classification project...")

    df = load_data("data/breast_cancer.csv")

    X, y = preprocess_data(df)
    
    # Split the data into train and test
    X_train, X_test, y_train, y_test = split_data(X, y)

    logreg_model = train_logistic_regression(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    evaluate_model(logreg_model, X_test, y_test, "Logistic Regression")
    evaluate_model(svm_model, X_test, y_test, "Support Vector Machine")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    logger.info("Project finished successfully!")

if __name__ == "__main__":
    main()
