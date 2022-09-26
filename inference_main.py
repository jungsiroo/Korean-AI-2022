import torch
import os
import warnings
import json
import glob
import pickle

from modules.eff_conformer.functions import *
from modules.eff_conformer.utils.preprocessing import * 
from modules.eff_conformer.arguments import get_args #custom
from modules.eff_conformer.arguments import *
from modules.inference import single_infer, single_infer_conformer

import nsml
from nsml import DATASET_PATH

def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        if model.tokenizer:
            with open(os.path.join(path, "tokenizer.pkl"), "wb") as f:
                pickle.dump(model.tokenizer, f)
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        if os.path.exists(os.path.join(path, "tokenizer.pkl")):
            with open(os.path.join(path, "tokenizer.pkl"), "rb") as f:
                model.tokenizer = pickle.load(f)

        if model.tokenizer:
            print("Tokenizer Loaded")

        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.

def inference(path, model, **kwargs):
    model.eval()
    print("INFERENCE ON")
    print(path)
    results = []
    for i in glob.glob(os.path.join(path, '*')):
        print(i)
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer_conformer(model, i)[0]
            }
        )
    return sorted(results, key=lambda x: x['filename'])

def main(args):
    # Device
    device = torch.device("cuda:" + str(args.rank) if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if hasattr(args, "num_threads") and int(args.num_threads) > 0:
        torch.set_num_threads(args.num_threads)

    # Load Config
    with open(args.config_file) as json_config:
        config = json.load(json_config)

    # Create Model
    model = create_model(config).to(device)
    # Load LM Config
    with open(config["decoding_params"]["lm_config"]) as json_config:
        config_lm = json.load(json_config)

        # Create LM
        model.lm = create_model(config_lm).to(device)

    bind_model(model,optimizer=model.optimizer)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode=="train":
        create_tokenizer(f"{os.getcwd()}/datasets", config["training_params"], config["tokenizer_params"])
        model = create_model(config).to(device)
        model.lm = create_model(config_lm).to(device)
        bind_model(model, optimizer=model.optimizer)
        nsml.load(3,session="KAIC312/t2-conf-final/459")
        nsml.save("EffConf")
        pass 
    # Summary
    model.summary(show_dict=args.show_dict)

    # Parallel Strategy
    if args.parallel and not args.distributed:
        print("Parallelize model on", torch.cuda.device_count(), "GPUs")
        model.parallel_strategy()


if __name__ == '__main__':
    args = get_args()
    warnings.filterwarnings('ignore')

    main(args)