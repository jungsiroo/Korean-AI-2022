import torch
import torch.nn as nn
import logging
import platform
from omegaconf import DictConfig

from modules.optim.lr_scheduler.lr_scheduler import LearningRateScheduler
from modules.vocabs import Vocabulary
from torch import optim
from modules.optim import (
    RAdam,
    AdamP,
    Novograd,
)
from modules.criterion import (
    LabelSmoothedCrossEntropyLoss,
    JointCTCCrossEntropyLoss,
    TransducerLoss,
)
from modules.optim.lr_scheduler import (
    TriStageLRScheduler,
    TransformerLRScheduler,
)


logger = logging.getLogger(__name__)


def check_envirionment(use_cuda: bool) -> torch.device:
    """
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return device


def get_optimizer(model: nn.Module, config: DictConfig):
    supported_optimizer = {
        'adam': optim.Adam,
        'radam': RAdam,
        'adamp': AdamP,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'novograd': Novograd,
    }
    assert config.optimizer.lower() in supported_optimizer.keys(), \
        f"Unsupported Optimizer: {config.optimizer}\n" \
        f"Supported Optimizer: {supported_optimizer.keys()}"

    if config.architecture == 'conformer':
        return optim.Adam(
            model.parameters(),
            betas=config.optimizer_betas,
            eps=config.optimizer_eps,
            weight_decay=config.weight_decay,
        )

    return supported_optimizer[config.optimizer](
        model.module.parameters(),
        lr=config.init_lr,
        weight_decay=config.weight_decay,
    )


def get_criterion(config: DictConfig, vocab: Vocabulary) -> nn.Module:
    if config.architecture in ('deepspeech2', 'jasper'):
        criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.reduction, zero_infinity=True)
    elif config.architecture in ('las', 'transformer') and config.joint_ctc_attention:
        criterion = JointCTCCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            reduction=config.reduction,
            ctc_weight=config.ctc_weight,
            cross_entropy_weight=config.cross_entropy_weight,
            blank_id=vocab.blank_id,
            dim=-1,
            smoothing=config.label_smoothing,
        )
    elif config.architecture == 'conformer':
        if config.decoder == 'rnnt':
            criterion = TransducerLoss(blank_id=vocab.blank_id)
        else:
            criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.reduction, zero_infinity=True)
    elif config.architecture == 'rnnt':
        criterion = TransducerLoss(blank_id=vocab.blank_id)
    elif config.architecture == 'transformer' and config.label_smoothing <= 0.0:
        criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_id,
            reduction=config.reduction,
        )
    else:
        criterion = LabelSmoothedCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            smoothing=config.label_smoothing,
            reduction=config.reduction,
            dim=-1,
        )

    return criterion


def get_lr_scheduler(config: DictConfig, optimizer, epoch_time_step) -> LearningRateScheduler:
    if config.lr_scheduler == "tri_stage_lr_scheduler":
        lr_scheduler = TriStageLRScheduler(
            optimizer=optimizer,
            init_lr=config.init_lr,
            peak_lr=config.peak_lr,
            final_lr=config.final_lr,
            init_lr_scale=config.init_lr_scale,
            final_lr_scale=config.final_lr_scale,
            warmup_steps=config.warmup_steps,
            total_steps=int(config.num_epochs * epoch_time_step),
        )
    elif config.lr_scheduler == "transformer_lr_scheduler":
        lr_scheduler = TransformerLRScheduler(
            optimizer=optimizer,
            peak_lr=config.peak_lr,
            final_lr=config.final_lr,
            final_lr_scale=config.final_lr_scale,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
        )
    else:
        raise ValueError(f"Unsupported Learning Rate Scheduler: {config.lr_scheduler}")

    return lr_scheduler

# import math
# import platform

# import torch
# import torch.nn as nn
# from torch.optim.optimizer import Optimizer
# from torch import optim

# from modules.vocab import Vocabulary


# class LearningRateScheduler(object):
#     """
#     Provides inteface of learning rate scheduler.

#     Note:
#         Do not use this class directly, use one of the sub classes.
#     """
#     def __init__(self, optimizer, init_lr):
#         self.optimizer = optimizer
#         self.init_lr = init_lr

#     def step(self, *args, **kwargs):
#         raise NotImplementedError

#     @staticmethod
#     def set_lr(optimizer, lr):
#         for g in optimizer.param_groups:
#             g['lr'] = lr

#     def get_lr(self):
#         for g in self.optimizer.param_groups:
#             return g['lr']


# class TriStageLRScheduler(LearningRateScheduler):
#     """
#     Tri-Stage Learning Rate Scheduler
#     Implement the learning rate scheduler in "SpecAugment"
#     """
#     def __init__(self, optimizer, init_lr, peak_lr, final_lr, init_lr_scale, final_lr_scale, warmup_steps, total_steps):
#         assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
#         assert isinstance(total_steps, int), "total_steps should be inteager type"

#         super(TriStageLRScheduler, self).__init__(optimizer, init_lr)
#         self.init_lr *= init_lr_scale
#         self.final_lr = final_lr
#         self.peak_lr = peak_lr
#         self.warmup_steps = warmup_steps
#         self.hold_steps = int(total_steps >> 1) - warmup_steps
#         self.decay_steps = int(total_steps >> 1)

#         self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
#         self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

#         self.lr = self.init_lr
#         self.update_step = 0

#     def _decide_stage(self):
#         if self.update_step < self.warmup_steps:
#             return 0, self.update_step

#         offset = self.warmup_steps

#         if self.update_step < offset + self.hold_steps:
#             return 1, self.update_step - offset

#         offset += self.hold_steps

#         if self.update_step <= offset + self.decay_steps:
#             # decay stage
#             return 2, self.update_step - offset

#         offset += self.decay_steps

#         return 3, self.update_step - offset

#     def step(self):
#         stage, steps_in_stage = self._decide_stage()

#         if stage == 0:
#             self.lr = self.init_lr + self.warmup_rate * steps_in_stage
#         elif stage == 1:
#             self.lr = self.peak_lr
#         elif stage == 2:
#             self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
#         elif stage == 3:
#             self.lr = self.final_lr
#         else:
#             raise ValueError("Undefined stage")

#         self.set_lr(self.optimizer, self.lr)
#         self.update_step += 1

#         return self.lr


class Optimizer(object):
    """
    This is wrapper classs of torch.optim.Optimizer.
    This class provides functionalities for learning rate scheduling and gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.Adam, torch.optim.SGD
        scheduler (modules.optim.lr_scheduler, optional): learning rate scheduler
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


# def get_lr_scheduler(config, optimizer, epoch_time_step) -> LearningRateScheduler:

#     lr_scheduler = TriStageLRScheduler(
#         optimizer=optimizer,
#         init_lr=config.init_lr,
#         peak_lr=config.peak_lr,
#         final_lr=config.final_lr,
#         init_lr_scale=config.init_lr_scale,
#         final_lr_scale=config.final_lr_scale,
#         warmup_steps=config.warmup_steps,
#         total_steps=int(config.num_epochs * epoch_time_step),
#     )

#     return lr_scheduler


# def get_optimizer(model: nn.Module, config):
#     supported_optimizer = {
#         'adam': optim.Adam,
#     }

#     return supported_optimizer[config.optimizer](
#         model.module.parameters(),
#         lr=config.init_lr,
#         weight_decay=config.weight_decay,
#     )


# def get_criterion(config, vocab: Vocabulary) -> nn.Module:

#     criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.reduction, zero_infinity=True)

#     return criterion