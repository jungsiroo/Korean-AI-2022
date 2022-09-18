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
from modules.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from modules.eff_conformer.functions import create_model
from modules.eff_conformer.arguments import get_args #custom
from modules.vocab import KoreanSpeechVocabulary
from modules.data import split_dataset, collate_fn
from modules.metrics import get_metric
from modules.inference import single_infer

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


def run(args):
    device = 'cuda' if args.use_cuda == True else 'cpu'
    if hasattr(args, "num_threads") and int(args.num_threads) > 0:
        torch.set_num_threads(args.num_threads)

    # Load Config
    with open(args.config_file) as json_config:
        config = json.load(json_config)

    # Model
    model = create_model(config).to(device)
    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    bind_model(model, optimizer=model.optimizer)

    metric = get_metric(metric_name='CER', vocab=vocab)

    # Summary
    model.summary(show_dict=args.show_dict)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == 'train':

        args.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
        preprocessing(label_path, os.getcwd(), args)
        train_dataset, valid_dataset = split_dataset(args, os.path.join(os.getcwd(), 'transcripts.txt'), vocab)

        model.fit(
            train_dataset, 
            config["training_params"]["epochs"], 
            dataset_val=valid_dataset, 
            val_steps=args.val_steps, 
            verbose_val=args.verbose_val, 
            initial_epoch=int(args.initial_epoch), 
            callback_path=config["training_params"]["callback_path"], 
            steps_per_epoch=args.steps_per_epoch,
            mixed_precision=config["training_params"]["mixed_precision"],
            accumulated_steps=config["training_params"]["accumulated_steps"],
            saving_period=args.saving_period,
            val_period=args.val_period
        )

        if args.gready or model.beam_size is None:

            if args.rank == 0:
                print("Gready Search Evaluation")
            wer, _, _, _ = model.evaluate(valid_dataset, eval_steps=args.val_steps, verbose=args.verbose_val, beam_size=1, eval_loss=args.eval_loss)
            
            if args.rank == 0:
                print("Geady Search WER : {:.2f}%".format(100 * wer))
        
        # Beam Search Evaluation
        else:

            if args.rank == 0:
                print("Beam Search Evaluation")
            wer, _, _, _ = model.evaluate(valid_dataset, eval_steps=args.val_steps, verbose=args.verbose_val, beam_size=model.beam_size, eval_loss=False)
            
            if args.rank == 0:
                print("Beam Search WER : {:.2f}%".format(100 * wer)) 

if __name__ == '__main__':
    args = get_args()
    warnings.filterwarnings('ignore')

    seed_fix(args.seed)
    run(args)