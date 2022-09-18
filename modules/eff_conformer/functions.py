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
from modules.eff_conformer.transducer import Transducer
from modules.eff_conformer.model_ctc import ModelCTC, InterCTC

def create_model(config):

    # Create Model
    if config["model_type"] == "Transducer":

        model = Transducer(
            encoder_params=config["encoder_params"],
            decoder_params=config["decoder_params"],
            joint_params=config["joint_params"],
            tokenizer_params=config["tokenizer_params"],
            training_params=config["training_params"],
            decoding_params=config["decoding_params"],
            name=config["model_name"]
        )

    elif config["model_type"] == "CTC":

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