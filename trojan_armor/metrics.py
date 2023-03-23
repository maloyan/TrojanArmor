# metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred, average="macro"):
        return precision_score(y_true, y_pred, average=average)

    @staticmethod
    def recall(y_true, y_pred, average="macro"):
        return recall_score(y_true, y_pred, average=average)

    @staticmethod
    def f1_score(y_true, y_pred, average="macro"):
        return f1_score(y_true, y_pred, average=average)

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        return confusion_matrix(y_true, y_pred, labels=labels)

