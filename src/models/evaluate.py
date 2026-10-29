from sklearn.metrics import recall_score, confusion_matrix, classification_report, average_precision_score

def evaluate_model(model, X_test, y_test):
    predicts = model.predict(X_test)
    print(classification_report(y_test,predicts))
    print(confusion_matrix(y_test,predicts))
    print(f"{recall_score(y_test, predicts):3f}")
    print(f"PR-AUC: {average_precision_score(y_test,predicts):3f}")