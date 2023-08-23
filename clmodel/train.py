import os
from time import time
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .evaluate import Metrics, evaluate
from .utils import AverageValue, Logger


def train(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
          optimizer: Optimizer, device: str, scaler: GradScaler = None,
          preprocess: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
          verbose: Optional[int] = None, logger: Logger = Logger()) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    This function trains a model for one epoch
    Args:
        model (nn.Module):          the model to be trained
        data_loader (DataLoader):   DataLoader for training
        criterion (nn.Molude):      loss function such as nn.CrossEntropyLoss
        optimizer (Optimizer):      optimizer such as adam
        device (str):               "cpu" or "cuda"
        scaler (GradScaler):        scaler for autocast
        preprocess (function):      the processing function for the model output layer such as sigmoid function
        verbose (int|None):         the frequency of the message printing
        logger (Logger):            logger of the training
    Returns:
        loss_average(float):        average training loss
        predictions(np.ndarray):    all the prediction result in this epoch in order
        labels(np.ndarray):         all the labels of the trained data used in this epoch in order
    """
    losses = AverageValue()
    model.train()
    model.to(device)
    start = time()
    predictions = []        # restore the prediction result in the training step
    labels = []             # restore the labels of the trained data used in the training step in order

    # training
    for step, (X, y) in enumerate(data_loader):
        # using autocast (reduce the GPU memory consumption)
        if scaler:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.cuda.amp.autocast():
                pred = model(X)
                if type(pred) == torch.Tensor:
                    y = y.type_as(pred)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.update(loss.item(), batch_size)
        # without autocast
        else:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), batch_size)
        
        predictions.append(preprocess(pred).to("cpu").detach().numpy())
        labels.append(y.to("cpu").detach().numpy())
        
        # printing and logging messages
        if verbose:
            if step % verbose == 0 or step == 0 or step == len(data_loader) - 1:
                now = time()
                rest_time = (now - start) * \
                    (len(data_loader) - step - 1) / (step + 1)
                message = f'Step: {step + 1}/{len(data_loader)} ' + \
                    f'Loss: {losses.avg:.4f} ' + \
                    f'Elapsed time {now - start:.1f} ' + \
                    f'Rest time {rest_time:.1f}'
                logger.write(message)

    predictions_res = np.concatenate(predictions, axis=0)
    labels_res = np.concatenate(labels, axis=0)
    return losses.avg, predictions_res, labels_res


def train_loop(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, fold_i: int, criterion: nn.Module,
                optimizer: Optimizer, device: str, n_epochs: int, scheduler: _LRScheduler = None, metrics: List[Metrics] = [],
                save_dir: Optional[str] = None, model_name: Optional[str] = None, preprocess=lambda x: x,
                verbose: Optional[int] = None, logger: Logger = Logger()) -> Tuple[dict, dict, dict, dict]:
    """
    This function manages the train step
    Args:
        model (nn.Module):          the model to be trained
        train_loader (DataLoader):  DataLoader for training
        valid_loader (DataLoader):  DataLoader for validation
        fold_i (int):               the fold of the current validation data
        criterion (nn.Molude):      loss function such as nn.CrossEntropyLoss
        optimizer (Optimizer):      optimizer such as adam
        device (str):               "cpu" or "cuda"
        n_epochs (int):             the number of epochs
        scheduler (_LRScheduler):   learning rate scheduler
        metrics (List[Metrics]):    the list of metrics to evaluate the trained model 
        save_dir (str|None):        the path to the directory to save the trained model in
        model_name (str|None):      the model_name
        preprocess (function):      the processing function for the model output layer such as sigmoid function
        verbose (int|None):         the frequency of the message printing
        logger (Logger):            logger of the training
    Returns:
        train_history (dict):       the training scores of each metrics
        valid_history (dict):       the validation scores of each metrics
        best_score (dict):          the best score in all the validation steps of each metrics
        best_predictions (dict):    the prediction result of the epoch in which you obtained the best validation score
    """
    metrics.append(Metrics("loss", None, "-"))          # Add the loss function as a metrics
    train_history: dict = {mtr.name: [] for mtr in metrics}
    valid_history: dict = {mtr.name: [] for mtr in metrics}
    best_score: dict = {
        mtr.name: (-np.inf if mtr.sign == "+" else np.inf) for mtr in metrics}      # in some metrics, the greater score is better
                                                                                    # but int others, the smaller is better
    best_predictions: dict = {mtr.name: None for mtr in metrics}
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, n_epochs+1):
        logger.write(f"Epoch {epoch}")
        if train_loader.dataset.cache_mode:
            train_loader.dataset.set_epoch(epoch-1)
        # training step
        loss, predictions, labels = train(
            model, train_loader, criterion, optimizer, device, scaler, preprocess, verbose, logger)
        if scheduler:
            scheduler.step()

        for mtr in metrics:
            if mtr.name == "loss":
                score = loss
            else:
                score = mtr.func(labels, predictions)
            train_history[mtr.name].append(score)

        # validation step
        loss, predictions, labels = evaluate(
            model, valid_loader, criterion, device, preprocess, verbose, logger)
        for mtr in metrics:
            if mtr.name == "loss":
                score = loss
            else:
                score = mtr.func(labels, predictions)
            valid_history[mtr.name].append(score)

            logger.write(f"{mtr.name} : {np.round(score, decimals=4)}")

            # update the best score
            if mtr.sign == "+" and score > best_score[mtr.name]:
                best_score[mtr.name] = score
                best_predictions[mtr.name] = (predictions, epoch)
                logger.write(f"This is best {mtr.name}.")
                if save_dir:
                    torch.save(model.to('cpu').state_dict(),
                               os.path.join(save_dir,
                                            f"{model_name}_fold{fold_i}_best_{mtr.name}.pth"))
                    logger.write(f"saved model.")
            elif mtr.sign == "-" and score < best_score[mtr.name]:
                best_score[mtr.name] = score
                best_predictions[mtr.name] = (predictions, epoch)
                logger.write(f"This is best {mtr.name}.")
                if save_dir:
                    torch.save(model.to('cpu').state_dict(),
                               os.path.join(save_dir,
                                            f"{model_name}_fold{fold_i}_best_{mtr.name}.pth"))
                    logger.write(f"saved model.")

    return train_history, valid_history, best_score, best_predictions


def visualize(train_history: dict, valid_history: dict, metrics: List[Metrics], multilabel: Optional[List[str]] = None,
            save_dir: str = None) -> None:
    """
    visualize the training results
    train_history (dict):           the scores in the training step
    valid_history (dict):           the scores in the validation step
    metrics (List[Metrics]):        metrics to be visualized
    multilabel (List[str]|None):    Given the label list, this function creates the figures for each label
    save_dir (str):                 the path to the directory to save the created figures in
    """
    n_epochs = max([len(hist) for name, hist in valid_history.items()])
    for mtr in metrics:
        if mtr.sign != "multi":
            hist = valid_history[mtr.name]
            if mtr.limit:
                plt.ylim(mtr.limit[0], mtr.limit[1])
            plt.plot(range(1, n_epochs+1), hist, label="valid")
            if mtr.name in train_history:
                plt.plot(range(1, n_epochs+1),
                         train_history[mtr.name], label="train")
            plt.title(mtr.name, fontsize=15)
            plt.legend()
            plt.xlabel("epoch")
            if save_dir:
                plt.savefig(os.path.join(
                    save_dir, f"{mtr.name}.png"), bbox_inches="tight")
            plt.close()
        elif multilabel:
            hist = valid_history[mtr.name]
            hist = [[hist[i][j]
                     for i in range(n_epochs)] for j in range(len(hist[0]))]
            for i, label_name in enumerate(multilabel):
                if mtr.limit:
                    plt.ylim(mtr.limit[0], mtr.limit[1])
                plt.plot(range(1, n_epochs+1), hist[i])
                plt.title(f"{mtr.name}:{label_name}", fontsize=15)
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f"{mtr.name}_{label_name}.png"),
                                bbox_inches="tight")
                plt.close()
            