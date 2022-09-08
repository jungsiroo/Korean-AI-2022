import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
import math
from glob import glob
import numpy as np

from modules.preprocess import preprocessing
from modules.conformer_lstm.trainer import trainer
from modules.conformer_lstm.utils import (
    get_conformer_optimizer,
    get_conformer_criterion,
    get_transformer_lr_scheduler,
)
from modules.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from modules.conformer_lstm.conformer_lstm import ConformerLSTMModel
from modules.vocab import KoreanSpeechVocabulary
from modules.data import split_dataset, collate_fn
from modules.conformer_lstm.utils import ConformerOptimizer
from modules.metrics import get_metric
from modules.inference import single_infer
from modules.conformer_lstm.arguments import get_args #custom

from torch.utils.data import DataLoader

import nsml
from nsml import DATASET_PATH

def seed_fix(seed): #시드 고정 함수
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.


def inference(path, model, **kwargs):
    model.eval()

    results = []
    for i in glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(model, i)[0]
            }
        )
    return sorted(results, key=lambda x: x['filename'])


def run(config):
    device = 'cuda' if config.use_cuda == True else 'cpu'
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)

    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    model = ConformerLSTMModel(config, vocab).to(device)

    optimizer = get_conformer_optimizer(model, config)
    bind_model(model, optimizer=optimizer)

    metric = get_metric(metric_name='CER', vocab=vocab)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':

        config.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
        preprocessing(label_path, os.getcwd(), config)
        train_dataset, valid_dataset = split_dataset(config, os.path.join(os.getcwd(), 'transcripts.txt'), vocab)

        lr_scheduler = get_transformer_lr_scheduler(config, optimizer, math.ceil(len(train_dataset)//config.train_batch_size))
        optimizer = ConformerOptimizer(optimizer, lr_scheduler, math.ceil(len(train_dataset)//config.train_batch_size)*config.num_epochs, config.max_grad_norm)
        criterion = get_conformer_criterion(config, vocab)

        num_epochs = config.num_epochs
        num_workers = config.num_workers

        train_begin_time = time.time()

        for epoch in range(num_epochs):
            print('[INFO] Epoch %d start' % epoch)

            # train

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=config.num_workers
            )

            model, train_loss, train_cer = trainer(
                'train',
                config,
                train_loader,
                optimizer,
                model,
                criterion,
                metric,
                train_begin_time,
                device
            )

            print('[INFO] Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

            # valid

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.valid_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=config.num_workers
            )

            model, valid_loss, valid_cer = trainer(
                'valid',
                config,
                valid_loader,
                optimizer,
                model,
                criterion,
                metric,
                train_begin_time,
                device
            )

            print('[INFO] Epoch %d (Validation) Loss %0.4f  CER %0.4f' % (epoch, valid_loss, valid_cer))

            nsml.report(
                summary=True,
                epoch=epoch,
                train_loss=train_loss,
                train_cer=train_cer,
                step=epoch*len(train_loader),
                lr = optimizer.get_lr(),
                val_loss=valid_loss,
                val_cer=valid_cer
            )

            if epoch % config.checkpoint_every == 0:
                nsml.save(epoch)

            torch.cuda.empty_cache()
            print(f'[INFO] epoch {epoch} is done')
        print('[INFO] train process is done')


if __name__ == '__main__':
    config = get_args()
    warnings.filterwarnings('ignore')

    seed_fix(config.seed)
    run(config)