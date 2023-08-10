import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from hn_cnn.constants import *

def get_accuracy(outputs, labels, threshold=0.5):
    """ Calculate the accuracy
    """
    preds = (outputs > threshold).int()[:,0]
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def compute_ci(pred_prob, labels, samples=1000, sig_level=0.05):
    """ Calculate the confidence interval
    """
    bootstrapped_aucs = []
    # bootstrapped_acc = []
    while len(bootstrapped_aucs) < samples:
    # for i in range(samples):
        indices = np.random.randint(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        auc = roc_auc_score(labels[indices], pred_prob[indices])
        # acc = get_accuracy(pred_prob[indices], labels[indices])   
        bootstrapped_aucs.append(auc)
        # bootstrapped_acc.append(acc)

    # print('CI')
    # print(np.sort(bootstrapped_aucs)[int(samples*sig_level/2) - 1])
    # print(np.sort(bootstrapped_aucs)[int(samples*(1-sig_level/2)) - 1])
    # print(np.mean(bootstrapped_aucs))
    # print(np.median(bootstrapped_aucs))

    # print('ACC')
    # print(np.sort(bootstrapped_acc)[int(samples*sig_level/2) - 1])
    # print(np.sort(bootstrapped_acc)[int(samples*(1-sig_level/2)) - 1])
    # print(np.mean(bootstrapped_acc))
    # print(np.median(bootstrapped_acc))
    return {
        MEAN: np.mean(bootstrapped_aucs),
        MEDIAN: np.median(bootstrapped_aucs),
        HIGHER_BOUND: np.sort(bootstrapped_aucs)[int(samples*(1-sig_level/2)) - 1],
        LOWER_BOUND: np.sort(bootstrapped_aucs)[int(samples*sig_level/2) - 1],
    }

def get_threshold(fprs, tprs, thresholds):
    """ Calculate the threshold from the AUC based on the 
        geometric mean
    """
    gmeans = []
    for i in range(len(thresholds)):
        gmeans.append(np.sqrt(tprs[i] * (1-fprs[i])))
    return np.argmax(gmeans)

def get_threshold_pr(precisions, recalls, thresholds):
    """ Calculate the threshold from the Precision-Recall curve
        based on the geometric mean
    """
    gmeans = []
    for i in range(len(thresholds)):
        div = (precisions[i] + recalls[i])
        if div == 0:
            gmeans.append(0)
        else:
            gmeans.append((2 * precisions[i] * recalls[i]) / div)
    return np.argmax(gmeans)
