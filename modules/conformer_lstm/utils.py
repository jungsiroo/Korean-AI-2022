import math
import platform
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch import optim
from typing import Tuple, Optional

class ConformerOptimizer(object):
    """
    This is wrapper classs of torch.optim.Optimizer.
    This class provides functionalities for learning rate scheduling and gradient norm clipping.
    Args:
        optim (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.Adam, torch.optim.SGD
        scheduler (kospeech.optim.lr_scheduler, optional): learning rate scheduler
        scheduler_period (int, optional): timestep with learning rate scheduler
        max_grad_norm (int, optional): value used for gradient norm clipping
    """
    def __init__(self, optim, scheduler=None, scheduler_period=None, max_grad_norm=0):
        self.optimizer = optim
        self.scheduler = scheduler
        self.scheduler_period = scheduler_period
        self.max_grad_norm = max_grad_norm
        self.count = 0

    def step(self, model):
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.scheduler is not None:
            self.update()
            self.count += 1

            if self.scheduler_period == self.count:
                self.scheduler = None
                self.scheduler_period = 0
                self.count = 0

    def set_scheduler(self, scheduler, scheduler_period):
        self.scheduler = scheduler
        self.scheduler_period = scheduler_period
        self.count = 0

    def update(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass
        else:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

def get_transformer_lr_scheduler(config, optimizer, epoch_time_step):
    lr_scheduler = TransformerLRScheduler(
        optimizer, 
        config.peak_lr / np.sqrt(config.encoder_dim), 
        config.final_lr, 
        config.final_lr_scale,
        config.warmup_steps,
        config.num_epochs * epoch_time_step - config.warmup_steps)
    return lr_scheduler

def get_conformer_optimizer(model: nn.Module, config):
    supported_optimizer = {
        'adam': optim.Adam
    }
    return supported_optimizer[config.optimizer](
        model.parameters(),
        lr=config.init_lr,
        weight_decay=config.weight_decay,
    )

class JointCTCCrossEntropyLoss(nn.Module):
    """
    Privides Joint CTC-CrossEntropy Loss function
    Args:
        num_classes (int): the number of classification
        ignore_index (int): indexes that are ignored when calculating loss
        dim (int): dimension of calculation loss
        reduction (str): reduction method [sum, mean] (default: mean)
        ctc_weight (float): weight of ctc loss
        cross_entropy_weight (float): weight of cross entropy loss
        blank_id (int): identification of blank for ctc
    """
    def __init__(
            self,
            num_classes: int,
            ignore_index: int,
            dim: int = -1,
            reduction='mean',
            ctc_weight: float = 0.3,
            cross_entropy_weight: float = 0.7,
            blank_id: int = None,
    ) -> None:
        super(JointCTCCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()
        self.ctc_weight = ctc_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=self.reduction, zero_infinity=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)

    def forward(
            self,
            encoder_log_probs,
            decoder_log_probs,
            output_lengths,
            targets,
            target_lengths,
    ):
        ctc_loss = self.ctc_loss(encoder_log_probs, targets, tuple(output_lengths), tuple(target_lengths))

        cross_entropy_loss = self.cross_entropy_loss(decoder_log_probs, targets.contiguous().view(-1))
 
        loss = cross_entropy_loss * self.cross_entropy_weight + ctc_loss * self.ctc_weight

        return loss

class LearningRateScheduler(object):
    """
    Provides inteface of learning rate scheduler.
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

class TransformerLRScheduler(LearningRateScheduler):
    def __init__(self, optimizer, peak_lr, final_lr, final_lr_scale, warmup_steps, decay_steps):
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(decay_steps, int), "total_steps should be inteager type"

        super(TransformerLRScheduler, self).__init__(optimizer, 0.0)
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_step = 0

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        if self.warmup_steps <= self.update_step < self.warmup_steps + self.decay_steps:
            return 1, self.update_step - self.warmup_steps

        return 2, None

    def step(self):
        self.update_step += 1
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_step * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)

        return self.lr

def get_conformer_criterion(config, vocab) -> nn.Module:
    criterion = JointCTCCrossEntropyLoss(
            num_classes=len(vocab.vocab_dict.keys()),
            ignore_index=vocab.pad_id,
            reduction="mean",
            blank_id=vocab.blank_id,
            dim=-1,
            cross_entropy_weight=0.7,
            ctc_weight=0.3,
        )

    return criterion