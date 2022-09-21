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

    dataset_train =  AIHubDataset(
        f"{DATASET_PATH}/train/train_data/",
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

    # Evaluation Dataset
    if  evaluation_split:

        # Multiple Evaluation datasets
        if isinstance(evaluation_split, list):

            dataset_eval = {}

            for split in evaluation_split:

                if args.rank == 0:
                    print("Loading evaluation dataset : {} {}".format(training_params["evaluation_dataset"], split))

                dataset = AIHubDataset(
                    training_params["evaluation_dataset_path"], 
                    training_params, 
                    tokenizer_params, 
                    split, 
                    args
                )

                if args.distributed:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size,rank=args.rank)
                else:
                    sampler = None

                dataset = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_eval, shuffle=(not args.distributed), num_workers=args.num_workers, collate_fn=collate_fn_pad, sampler=sampler, pin_memory=False)
                
                if args.rank == 0:
                    print("Loaded :", dataset.dataset.__len__(), "samples", "/", dataset.__len__(), "batches")

                dataset_eval[split] = dataset

        # One Evaluation dataset
        else:

            if args.rank == 0:
                print("Loading evaluation dataset : {} {}".format(training_params["evaluation_dataset"], evaluation_split))

            dataset_eval = AIHubDataset(training_params["evaluation_dataset_path"], training_params, tokenizer_params, evaluation_split, args)

            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval, num_replicas=args.world_size,rank=args.rank)
            else:
                sampler = None

            dataset_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size_eval, shuffle=(not args.distributed), num_workers=args.num_workers, collate_fn=collate_fn_pad, sampler=sampler, pin_memory=False)
            
            if args.rank == 0:
                print("Loaded :", dataset_eval.dataset.__len__(), "samples", "/", dataset_eval.__len__(), "batches")
    else:
        dataset_eval = None
    
    return dataset_train, dataset_eval