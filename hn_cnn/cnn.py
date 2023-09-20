import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F

from hn_cnn.constants import *
from hn_cnn.metrics import get_accuracy, get_threshold, get_threshold_pr, compute_ci

class ImageClassificationBase(nn.Module):
    """ CNN base class extension
    """
    def training_step(self, batch, class_weights):
        """ Perform a training step
        """
        # Retieve data
        images, tab, labels = batch
        # Generate predictions
        out = self(images, tab)
        loss = F.binary_cross_entropy(
            out[:, 0],
            labels.float(),
            weight=torch.where(labels == 0, float(class_weights[0]), float(class_weights[1]))
        )
        return loss


    def validation_step(self, batch, class_weights):
        """ Perform a validation step
        """
        # Generate predictions
        images, clinical_data, labels = batch
        predictions = self(images, clinical_data)
        # Calculate the loss
        # loss = self.loss_func(out[:, 0], labels.float())
        loss = F.binary_cross_entropy(
            predictions[:, 0],
            labels.float(),
            weight=torch.where(labels == 0, float(class_weights[0]), float(class_weights[1]))
        )
        # Calculate accuracy
        accuracy = get_accuracy(predictions, labels)
        # ROC curve and respective AUC
        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        threshold = get_threshold(fpr, tpr, thresholds)
        accuracy_roc = get_accuracy(predictions, labels, threshold=thresholds[threshold]) 
        # Precision-Recall curve and respective AUC
        precision, recall, thresholds_pr = metrics.precision_recall_curve(labels, predictions)
        auc_pr = -np.trapz(precision, recall)
        pr_score = metrics.average_precision_score(labels, predictions, pos_label=1)
        threshold_pr = get_threshold_pr(precision, recall, thresholds_pr)
        accuracy_pr = get_accuracy(predictions, labels, threshold=thresholds_pr[threshold_pr]) 
        # Return the metrics
        performance = {
            LOSS: loss.detach().numpy(),
            ACCURACY: accuracy,
            ROC: {
                AUC: auc,
                ACCURACY: accuracy_roc,
                THRESHOLD: thresholds[threshold],
                TPR: tpr[threshold],
                FPR: fpr[threshold],
            },
            PR: {
                ACCURACY: accuracy_pr,
                AUC: auc_pr,
                SCORE: pr_score,
                THRESHOLD: thresholds_pr[threshold_pr],
                RECALL: recall[threshold_pr],
                PRECISION: precision[threshold_pr],
            }

        }
        return performance

class HNCNN(ImageClassificationBase):
    """ Implementation of the CNN to predict outcomes
    """
    def __init__(self, base_number_filters=8, leaky_relu_slope=0.01, clinical_data=False):
        super().__init__()
        #self.clinical_data = len(clinical_data) > 0
        self.clinical_data = clinical_data
        # self.lr = nn.ReLU()
        self.lr = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        # Convolutional layer 1
        self.cv1 = nn.Conv2d(1, base_number_filters, (3,3), stride=1)
        self.mxp1 = nn.MaxPool2d((3, 3), stride=1)
        # Convolutional layer 2
        self.cv2 = nn.Conv2d(base_number_filters, 2 * base_number_filters, (3,3), stride=2)
        self.mxp2 = nn.MaxPool2d((3, 3), stride=2)
        # Convolutional layer 3
        self.cv3 = nn.Conv2d(2 * base_number_filters, 4 * base_number_filters, (4, 4), stride=3)
        self.mxp3 = nn.MaxPool2d((4, 4), stride=3)
        # Fully connected Layers
        self.ln1 = nn.Linear(64 * base_number_filters, 16 * base_number_filters)
        self.dp1 = nn.Dropout(0.3)
        self.ln2 = nn.Linear(16 * base_number_filters, 8 * base_number_filters)
        self.dp2 = nn.Dropout(0.2)
        self.ln3 = nn.Linear(8 * base_number_filters, 4 * base_number_filters)
        self.dp3 = nn.Dropout(0.1)
        # Last activation layer
        #clinical_neurons = len(clinical_data)
        clinical_neurons = 11 if clinical_data else 0
        self.ln4 = nn.Linear(4 * base_number_filters + clinical_neurons, 1)
        self.act = nn.Sigmoid()

    def forward(self, img, tab=[]):
        # Covolutional blocks
        out = self.lr(self.mxp1(self.cv1(img)))
        out = self.lr(self.mxp2(self.cv2(out)))
        out = self.lr(self.mxp3(self.cv3(out)))
        # Flatten the features
        out = torch.flatten(out, 1)
        # Fully connected layers
        out = self.dp1(self.lr(self.ln1(out)))
        out = self.dp2(self.lr(self.ln2(out)))
        out = self.dp3(self.lr(self.ln3(out)))
        if self.clinical_data:
            out = torch.concat((out, tab), 1)
        out = self.act(self.ln4(out))
        return out

class HNANN(ImageClassificationBase):
    """ Implementation of the ANN to predict outcomes
    """
    def __init__(self, leaky_relu_slope=0.01, clinical_data=None):
        super().__init__()
        # self.lr = nn.ReLU()
        #clinical_neurons = len(clinical_data)
        clinical_neurons = 11
        self.lr = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        # Fully connected Layers
        self.ln1 = nn.Linear(clinical_neurons, 8)
        self.dp1 = nn.Dropout(0.125)
        self.ln2 = nn.Linear(8, 4)
        self.dp2 = nn.Dropout(0.1)
        self.ln3 = nn.Linear(4, 1)
        self.act = nn.Sigmoid()

    def forward(self, img, tab=[]):
        out = self.dp1(self.lr(self.ln1(tab)))
        out = self.dp2(self.lr(self.ln2(out)))
        out = self.act(self.ln3(out))
        return out

class HNLR():
    """ Implementation of the Logistic Regression model
    """
    def __init__(self, clinical_data=None):
        self.model = LogisticRegression()

    def training(self, batch, regularization=10**-2, class_weights=[1.0, 1.0]):
        """ Train the model
        """
        self.model.C = regularization
        self.model.class_weight = {0: class_weights[0], 1: class_weights[1]}
        # Retrieve the data
        _, clinical_data, labels = batch
        # Train model
        return self.model.fit(clinical_data, labels)

    def validation_step(self, batch):
        """ Calculate the metrics for a batch
        """
        results = {}
        # Retrieve the data
        _, clinical_data, labels = batch
        # Predict and compute metrics
        # predictions = self.model.predict(clinical_data)
        predicted_prob = self.model.predict_proba(clinical_data)[:,1] 
        # Metrics
        results[ACCURACY] = self.model.score(clinical_data, labels)
        results[AUC] = metrics.roc_auc_score(labels, predicted_prob)
        results[CI] = compute_ci(predicted_prob, labels, samples=1000, sig_level=0.05)
        return results


# Map for the models
MODELS = {
    CNN: HNCNN,
    ANN: HNANN,
    LR: HNLR,
}
