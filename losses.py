import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def loss_fn(X, X_pred):
    loss=nn.MSELoss(X, X_pred)
    return loss


