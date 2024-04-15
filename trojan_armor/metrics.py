# metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

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

    @staticmethod
    def clean_accuracy(y_true_clean, y_pred_clean):
        return accuracy_score(y_true_clean, y_pred_clean)

    @staticmethod
    def attack_success_rate(y_true_poisoned, y_pred_poisoned, target_label):
        successful_attacks = np.sum((y_pred_poisoned == target_label) & (y_true_poisoned != target_label))
        total_attacks = np.sum(y_true_poisoned != target_label)
        return successful_attacks / total_attacks

    @staticmethod
    def attack_deduction_rate(y_true_poisoned, y_pred_poisoned_original, y_pred_poisoned_defended, target_label):
        asr_before_defense = Metrics.attack_success_rate(y_true_poisoned, y_pred_poisoned_original, target_label)
        asr_defended = Metrics.attack_success_rate(y_true_poisoned, y_pred_poisoned_defended, target_label)
        return asr_before_defense - asr_defended

    @staticmethod
    def robust_accuracy(y_true_poisoned, y_pred_poisoned):
        return accuracy_score(y_true_poisoned, y_pred_poisoned)

    @staticmethod
    def trojan_misclassification_confidence(y_pred_poisoned, target_label):
        target_probs = y_pred_poisoned[y_pred_poisoned.argmax(axis=1) == target_label].max(axis=1)
        return np.mean(target_probs)

    @staticmethod
    def clean_accuracy_drop(y_true_clean, y_pred_clean, y_true_clean_defended, y_pred_clean_defended):
        clean_acc = Metrics.clean_accuracy(y_true_clean, y_pred_clean)
        clean_acc_defended = Metrics.clean_accuracy(y_true_clean_defended, y_pred_clean_defended)
        return clean_acc - clean_acc_defended

    @staticmethod
    def clean_classification_confidence(y_pred_clean):
        return np.mean(y_pred_clean.max(axis=1))

    @staticmethod
    def efficacy_specificity_auc(y_true, y_pred_scores):
        return roc_auc_score(y_true, y_pred_scores)

    @staticmethod
    def neuron_separation_ratio():
        # This metric is model-specific and depends on neuron activations.
        # You need to provide more details on how to calculate it.
        pass
