"""Evaluation utilities: metrics, confusion matrix, ROC plotting."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    return {'accuracy': acc, 'macro_f1': macro_f1, 'micro_f1': micro_f1, 'report': report}


def plot_confusion(y_true, y_pred, labels=None, figsize=(6,5), save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compute_roc_auc(y_true, y_probs, num_classes=None):
    # y_probs should be (n_samples, n_classes)
    if num_classes is None:
        num_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
    except Exception:
        auc = None
    return auc


if __name__ == '__main__':
    print('Evaluation utilities ready')
