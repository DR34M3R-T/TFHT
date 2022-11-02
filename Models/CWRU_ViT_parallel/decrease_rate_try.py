import types
import math
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class Decrease_inregular(_LRScheduler):
    """
    在epoch小于min_iter的时候让准确率在min_lr附近
    当到达min_iter的时候迅速(这里是直接直线)涨到max_lr
    当处于min_iter与max_iter的时候指数下降
    
        Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, min_lr=1e-6, max_lr=1e-2 ,need_iter=5, last_epoch=-1, verbose=False):
        if min_lr > 1.0 or min_lr < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if max_lr > 1.0 or max_lr < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.need_iter = need_iter
        self.gamma = gamma
        super(Decrease_inregular, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch < self.need_iter):
            return [group['lr'] for group in self.optimizer.param_groups]

        if (self.last_epoch == self.need_iter):
            return [group['lr'] * (self.max_lr/self.min_lr) for group in self.optimizer.param_groups]
        
        if (self.last_epoch > self.need_iter):
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]