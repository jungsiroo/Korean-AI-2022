import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
import math
import glob
import numpy as np


from modules.eff_conformer.functions import *
from modules.eff_conformer.utils.preprocessing import * 
from modules.eff_conformer.arguments import get_args #custom
from modules.inference import single_infer, single_infer_conformer

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
    for i in glob.glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer_conformer(model, i)[0]
            }
        )
    return sorted(results, key=lambda x: x['filename'])


def run(args):

    # Device
    device = torch.device("cuda:" + str(args.rank) if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if hasattr(args, "num_threads") and int(args.num_threads) > 0:
        torch.set_num_threads(args.num_threads)

    # Load Config
    with open(args.config_file) as json_config:
        config = json.load(json_config)

    # Create Tokenizer
    if args.create_tokenizer:
        if args.rank == 0:
            print("Creating Tokenizer")
            create_tokenizer(f"{os.getcwd()}/datasets", config["training_params"], config["tokenizer_params"])

        if args.distributed:
            torch.distributed.barrier()

    # Create Model
    model = create_model(config).to(device)

    # Load LM Config
    with open(config["decoding_params"]["lm_config"]) as json_config:
        config_lm = json.load(json_config)

        # Create LM
        model.lm = create_model(config_lm).to(device)

    # bind_model(model, optimizer=model.optimizer)

    # Summary
    model.summary(show_dict=args.show_dict)

    # Parallel Strategy
    if args.parallel and not args.distributed:
        print("Parallelize model on", torch.cuda.device_count(), "GPUs")
        model.parallel_strategy()

    # Prepare Dataset
    if args.prepare_dataset:
        if args.rank == 0:
            print("Preparing dataset")
            prepare_dataset(config["training_params"], config["tokenizer_params"], model.tokenizer)
            
    else:
        if args.rank == 0:
            print("Load Dataset from CSV")
            load_data_csv(os.path.join(os.getcwd(), "encode.csv"))

    # Load Dataset
    dataset_train, dataset_val = load_datasets(config["training_params"], config["tokenizer_params"], args)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == 'train':
        model.fit(
            dataset_train, 
            config["training_params"]["epochs"], 
            bind_model, 
            dataset_val=dataset_val, 
            val_steps=args.val_steps, 
            verbose_val=args.verbose_val, 
            initial_epoch=int(args.initial_epoch), 
            callback_path=config["training_params"]["callback_path"], 
            steps_per_epoch=None,
            mixed_precision=config["training_params"]["mixed_precision"],
            accumulated_steps=config["training_params"]["accumulated_steps"],
            saving_period=args.saving_period,
            val_period=args.val_period
        )

    if args.gready or model.beam_size is None:

        if args.rank == 0:
            print("Gready Search Evaluation")
        cer, _, _, _ = model.evaluate(dataset_val, eval_steps=args.val_steps, verbose=args.verbose_val, beam_size=1, eval_loss=args.eval_loss)
        
        if args.rank == 0:
            print("Geady Search CER : {:.2f}%".format(100 * cer))

    if args.distributed:
        torch.distributed.destroy_process_group()
        

if __name__ == '__main__':
    args = get_args()
    warnings.filterwarnings('ignore')

    seed_fix(args.seed)
    run(args)