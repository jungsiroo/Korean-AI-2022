# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch

# Models
from modules.eff_conformer.model_ctc import ModelCTC, InterCTC

# Datasets
from modules.eff_conformer.utils.datasets import (
    AIHubDataset,
)
import nsml
from nsml import DATASET_PATH

# Preprocessing
from modules.eff_conformer.utils.preprocessing import (
    collate_fn_pad
)

import glob

def create_model(config):

    if config["model_type"] == "CTC":

        model = ModelCTC(
            encoder_params=config["encoder_params"],
            tokenizer_params=config["tokenizer_params"],
            training_params=config["training_params"],
            decoding_params=config["decoding_params"],
            name=config["model_name"]
        )

    elif config["model_type"] == "InterCTC":

        model = InterCTC(
            encoder_params=config["encoder_params"],
            tokenizer_params=config["tokenizer_params"],
            training_params=config["training_params"],
            decoding_params=config["decoding_params"],
            name=config["model_name"]
        )

    else:

        raise Exception("Unknown model type")

    return model

def load_datasets(training_params, tokenizer_params, args):
    # Training Dataset
    if args.rank == 0:
        print("Loading training dataset : {} {}".format(training_params["training_dataset"], "For Training"))

    train_path = glob.glob(f"/app/datasets/train/*")
    pivot = int(len(train_path)*0.8)
    train_data = train_path[:pivot]
    valid_data = train_path[pivot:]

    dataset_train =  AIHubDataset(
        train_data,
        training_params, 
        tokenizer_params, 
        "train",
        args
    )

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=args.world_size,rank=args.rank)
    else:
        sampler = None

    dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=training_params["batch_size"], shuffle=(not args.distributed), num_workers=args.num_workers, collate_fn=collate_fn_pad, drop_last=True, sampler=sampler, pin_memory=False)
    
    if args.rank == 0:
        print("Loaded :", dataset_train.dataset.__len__(), "samples", "/", dataset_train.__len__(), "batches")

# One Evaluation dataset
    if args.rank == 0:
        print("Loading evaluation dataset : {} {}".format(training_params["evaluation_dataset"], "Validation"))

    dataset_eval = AIHubDataset(
        valid_data,
        training_params, 
        tokenizer_params, 
        "valid",
        args)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval, num_replicas=args.world_size,rank=args.rank)
    else:
        sampler = None

    dataset_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_pad, sampler=sampler, pin_memory=False)
    
    if args.rank == 0:
        print("Loaded :", dataset_eval.dataset.__len__(), "samples", "/", dataset_eval.__len__(), "batches")

    return dataset_train, dataset_eval