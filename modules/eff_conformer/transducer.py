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
import torch.nn as nn

# Base Model
from modules.eff_conformer.model import Model, init_vn

# Encoders
from modules.eff_conformer.encoders import (
    ConformerEncoder
)

# Decoders
from modules.eff_conformer.decoders import (
    RnnDecoder,
    TransformerDecoder,
    ConformerDecoder
)

# Joint Network
from modules.eff_conformer.joint_networks import (
    JointNetwork
)

# Language Model
from modules.eff_conformer.lm import (
    LanguageModel
)

# Losses
from modules.eff_conformer.losses import (
    LossRNNT
)

# Ngram
# import kenlm

class Transducer(Model):

    def __init__(self, encoder_params, decoder_params, joint_params, tokenizer_params, training_params, decoding_params, name):
        super(Transducer, self).__init__(tokenizer_params, training_params, decoding_params, name)

        # Encoder
        if encoder_params["arch"] == "Conformer":
            self.encoder = ConformerEncoder(encoder_params)
        else:
            raise Exception("Unknown encoder architecture:", encoder_params["arch"])

        # Decoder
        if decoder_params["arch"] == "RNN":
            self.decoder = RnnDecoder(decoder_params)
        elif decoder_params["arch"] == "Transformer":
            self.decoder = TransformerDecoder(decoder_params)
        elif decoder_params["arch"] == "Conformer":
            self.decoder = ConformerDecoder(decoder_params)
        else:
            raise Exception("Unknown decoder architecture:", decoder_params["arch"])

        # Joint Network
        self.joint_network = JointNetwork(encoder_params["dim_model"][-1] if isinstance(encoder_params["dim_model"], list) else  encoder_params["dim_model"], decoder_params["dim_model"], decoder_params["vocab_size"], joint_params)

        # Init VN
        self.decoder.apply(lambda m: init_vn(m, training_params.get("vn_std", None)))

        # Criterion
        self.criterion = LossRNNT()

        # Decoding
        self.max_consec_dec_step = decoder_params.get("max_consec_dec_step", 5)

        # Compile
        self.compile(training_params)

    def forward(self, batch):

        # Unpack Batch
        x, y, x_len, y_len = batch

        # Audio Encoder (B, Taud) -> (B, T, Denc)
        f, f_len, attentions = self.encoder(x, x_len)

        # Add blank token
        y = torch.nn.functional.pad(y, pad=(1, 0, 0, 0), value=0)
        y_len = y_len + 1

        # Text Decoder (B, U + 1) -> (B, U + 1, Ddec)
        g, _ = self.decoder(y, None, y_len)

        # Joint Network (B, T, Denc) and (B, U + 1, Ddec) -> (B, T, U + 1, V)
        logits = self.joint_network(f, g)

        return logits, f_len, attentions

    def distribute_strategy(self, rank):
        super(Transducer, self).distribute_strategy(rank)

        self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.rank])
        self.decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder)
        self.decoder = torch.nn.parallel.DistributedDataParallel(self.decoder, device_ids=[self.rank])
        self.joint_network = torch.nn.parallel.DistributedDataParallel(self.joint_network, device_ids=[self.rank])

    def parallel_strategy(self):
        super(Transducer, self).parallel_strategy()

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)
        self.joint_network = torch.nn.DataParallel(self.joint_network)

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters :", self.num_params() - self.lm.num_params() if isinstance(self.lm, LanguageModel) else self.num_params())
        print(" - Encoder Parameters :", sum([p.numel() for p in self.encoder.parameters()]))
        print(" - Decoder Parameters :", sum([p.numel() for p in self.decoder.parameters()]))
        print(" - Joint Parameters :", sum([p.numel() for p in self.joint_network.parameters()]))

        if isinstance(self.lm, LanguageModel):
            print("LM Parameters :", self.lm.num_params())

        if show_dict:
            for key, value in self.state_dict().items():
                print("{:<64} {:<16} mean {:<16.4f} std {:<16.4f}".format(key, str(tuple(value.size())), value.float().mean(), value.float().std()))

    def gready_search_decoding(self, x, x_len):

        # Predictions String List
        preds = []

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        f, f_len, _ = self.encoder(x, x_len)

        # Batch loop
        for b in range(x.size(0)): # One sample at a time for now, not batch optimized

            # Init y and hidden state
            y = x.new_zeros(1, 1, dtype=torch.long)
            hidden = None

            enc_step = 0
            consec_dec_step = 0

            # Decoder loop
            while enc_step < f_len[b]:

                # Forward Decoder (1, 1) -> (1, 1, Ddec)
                g, hidden = self.decoder(y[:, -1:], hidden)
                
                # Joint Network loop
                while enc_step < f_len[b]:

                    # Forward Joint Network (1, 1, Denc) and (1, 1, Ddec) -> (1, V)
                    logits = self.joint_network(f[b:b+1, enc_step], g[:, 0])

                    # Token Prediction
                    pred = logits.softmax(dim=-1).log().argmax(dim=-1) # (1)

                    # Null token or max_consec_dec_step
                    if pred == 0 or consec_dec_step == self.max_consec_dec_step:
                        consec_dec_step = 0
                        enc_step += 1
                    # Token
                    else:
                        consec_dec_step += 1
                        y = torch.cat([y, pred.unsqueeze(0)], axis=-1)
                        break

            # Decode Label Sequence
            pred = self.tokenizer.decode(y[:, 1:].tolist())
            preds += pred

        return preds
