from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def calc_auc_score(predictions, truth):
    """
    A simple function for calculating the AUC score.

    XXX: Do not change this function.
    """
    fpr, tpr, th = roc_curve(truth, predictions)
    return auc(fpr, tpr)
