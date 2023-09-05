from time import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from torch.utils.data import DataLoader

from .model import CLModel
from .utils import AverageValue, Logger


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    preprocess: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    verbose: Optional[int] = None,
    logger: Logger = Logger(),
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    This function evaluates a model.
    Args:
        model (nn.Module):          the model to be evaluated
        data_loader (DataLoader):   DataLoader for evaluation
        criterion (nn.Molude):      loss function such as nn.CrossEntropyLoss
        device (str):               "cpu" or "cuda"
        preprocess (function):      the processing function for the model output layer such as sigmoid function
        verbose (int|None):         the frequency of the message printing
        logger (Logger):            logger of the training
    Returns:
        loss_average(float):        average loss
        predictions(np.ndarray):    all the prediction result in this evaluation in order
        labels(np.ndarray):         all the labels of the trained data used in this evaluation in order
    """
    losses = AverageValue()
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    start = time()

    # evaluation
    for step, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        batch_size = y.size(0)
        with torch.no_grad():
            pred = model(X)
            if type(pred) == torch.Tensor:
                y = y.type_as(pred)
            loss = criterion(pred, y)
        losses.update(loss.item(), batch_size)
        predictions.append(preprocess(pred).to("cpu").numpy())
        labels.append(y.to("cpu").numpy())

        # printing and logging messages
        if verbose:
            if step % verbose == 0 or step == 0 or step == len(data_loader) - 1:
                now = time()
                rest_time = (now - start) * (len(data_loader) - step - 1) / (step + 1)
                message = (
                    f"Step: {step + 1}/{len(data_loader)} "
                    + f"Loss: {losses.avg:.4f} "
                    + f"Elapsed time {now - start:.1f} "
                    + f"Rest time {rest_time:.1f}"
                )
                logger.write(message)
    predictions_res = np.concatenate(predictions)
    labels_res = np.concatenate(labels)
    return losses.avg, predictions_res, labels_res


def extract_features(
    model: CLModel,
    data_loader: DataLoader,
    device: str,
    n_samples: int,
    verbose: Optional[int] = None,
    logger: Logger = Logger(),
) -> np.ndarray:
    """
    This function obtains features extracted with a model.
    Args:
        model (nn.Module):          The model for feature extraction.
        data_loader (DataLoader):   DataLoader for data from which the model is to extract features
        device (str):               "cpu" or "cuda"
        n_samples (int):            The number of feature extractions. Increasing this number will reduce the error caused by sampling
        verbose (int|None):         the frequency of the message printing
        logger (Logger):            logger of the training
    Outputs:
        features (np.ndarray):      the extracted features
    """
    features = np.array(0)
    for repeat in range(1, n_samples + 1):
        logger.write(f"Repeat {repeat}")
        model.to(device)
        model.eval()
        preds = []
        start = time()
        for step, (X, y) in enumerate(data_loader):
            X = X.to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                pred = model.forward_features(X)
            preds.append(pred.to("cpu").numpy())
            if verbose:
                if step % verbose == 0 or step == 0 or step == len(data_loader) - 1:
                    now = time()
                    rest_time = (
                        (now - start) * (len(data_loader) - step - 1) / (step + 1)
                    )
                    message = (
                        f"Step: {step + 1}/{len(data_loader)} "
                        + f"Elapsed time {now - start:.1f} "
                        + f"Rest time {rest_time:.1f}"
                    )
                    logger.write(message)

        _features = np.concatenate(preds)
        features += _features
    return features / n_samples


class Metrics(object):
    """
    This class restore metrics to evaluate models.
    Args:
        name (str):                         The model for feature extraction. You should remove the last FC layer.
        func (function|None):               DataLoader for data from which the model is to extract features
        sign (str):                         "+" or "-". If "+" is chosen, the higher score is better, viceversa the lower is better.
        limit (Tuple[float, float]|None):   The upper and lower limits. For example, 1 is the upper limit and 0 is the lower limit as for accuracy.
    """

    def __init__(
        self,
        name: str,
        func: Optional[Callable[[np.ndarray, np.ndarray], float]],
        sign: str,
        limit: Optional[Tuple[float, float]] = None,
    ) -> None:
        self._name = name
        self._func = func
        self._sign = sign  # ["+", "-", "multi"]
        self._limit = limit

    @property
    def name(self) -> str:
        return self._name

    @property
    def func(self) -> Optional[Callable[[np.ndarray, np.ndarray], float]]:
        return self._func

    @property
    def sign(self) -> str:
        return self._sign

    @property
    def limit(self) -> Optional[Tuple[float, float]]:
        return self._limit


def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    res = []
    for i in range(y_true.shape[1]):
        try:
            res.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except:
            res.append(np.nan)
    return np.array(res)


################################################################################
####################### functions for metrics ##################################
################################################################################


def macro_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.nanmean(auroc(y_true, y_pred))


def limited_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pair_info: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    return np.array([auroc(y_true[f, idx], y_pred[f, idx]) for f, idx in pair_info])


def limited_macro_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pair_info: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    return np.nanmean(limited_auroc(y_true, y_pred, pair_info))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.array(
        [np.mean(y_true[:, i] == (y_pred[:, i] > 0.5)) for i in range(y_true.shape[1])]
    )


def macro_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.nanmean(auroc(y_true, y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.array(
        [
            balanced_accuracy_score(y_true[:, i], (y_pred[:, i] > 0.5))
            for i in range(y_true.shape[1])
        ]
    )


# def mAP(y_true: np.ndarray, y_pred: np.ndarray) -> np.array:
#     pass

# def macro_mAP(y_true: np.ndarray, y_pred: np.ndarray) -> np.array:
#     pass

# def MCC(y_true: np.ndarray, y_pred: np.ndarray) -> np.array:
#     pass

# def macro_MCC(y_true: np.ndarray, y_pred: np.ndarray) -> np.array:
#     pass

def macro_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.nanmean(balanced_accuracy(y_true, y_pred))


def macro_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.nanmean(
        [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    )


mtr_dict = {
    "macro_auroc": {"name": "macro AUROC", "func": macro_auroc, "sign": "+", "limit": (0,1)},
    "auroc": {"name": "AUROC", "func": auroc, "sign": "multilabel", "limit": (0,1)},
    "macro_accuracy": {"name": "macro acc", "func": macro_accuracy, "sign": "+", "limit": (0,1)},
    "accuracy": {"name": "acc", "func": accuracy, "sign": "multilabel", "limit": (0,1)},
    "auroc": {"name": "AUROC", "func": auroc, "sign": "multilabel"}, #ここから
}


def metrics_resolver(mtr_name: str):
    if mtr_name == "macro_auroc":
        return Metrics("macro AUROC", macro_auroc, "+", limit=(0, 1))
    elif mtr_name == "auroc":
        return Metrics("AUROC", auroc, "multilabel", limit=(0, 1))
    elif mtr_name == "macro_accuracy":
        return Metrics("macro acc", macro_accuracy, "+", limit=(0, 1))
    elif mtr_name == "accuracy":
        return Metrics("acc", accuracy, "multilabel", limit=(0, 1))
    elif mtr_name == "macro_balanced_accuracy":
        return Metrics("macro balanced acc", macro_balanced_accuracy, "+", limit=(0, 1))
    elif mtr_name == "macro_r2_score":
        return Metrics("macro r2 score", macro_r2_score, "+", limit=(0, 1))


class WeightedBCELossWithLogits(nn.Module):
    """
    This class is for weighted binary cross entropy loss with logits.
    Args:
        positive_weight (torch.Tensor):     the weight of positive data
        negative_weight (torch.Tensor):     the weight of negative data
        device (str):                       "cpu" or "cuda"
        eps (float):                        eps
    """

    def __init__(
        self,
        positive_weight: torch.Tensor,
        negative_weight: torch.Tensor,
        device: str = "cpu",
        eps: float = 1e-6,
    ) -> None:
        self.positive_weight = positive_weight.float().to(device)
        self.negative_weight = negative_weight.float().to(device)
        self.eps = eps

    def __call__(self, y_preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        p = y_preds.sigmoid()
        return -torch.mean(
            self.positive_weight * labels * torch.log(p + self.eps)
            + self.negative_weight * (1 - labels) * torch.log(1 - p + self.eps)
        )


class FocalBCELossWithLogits(object):
    """
    This class is for focal binary cross entropy loss with logits.
    Args:
        gamma (float):          gamma
        device (str):           "cpu" or "cuda"
        eps (float):            eps
    """

    def __init__(
        self, gamma: float = 1.0, device: str = "cpu", eps: float = 1e-6
    ) -> None:
        self.gamma = gamma
        self.eps = eps

    def __call__(self, y_preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        p = y_preds.sigmoid()
        return -torch.mean(
            ((1 - p) ** self.gamma) * labels * torch.log(p + self.eps)
            + (p**self.gamma) * (1 - labels) * torch.log(1 - p + self.eps)
        )
