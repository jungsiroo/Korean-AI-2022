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
import torchaudio

# Sentencepiece
import sentencepiece as spm

# Other
import glob
from tqdm import tqdm

# Librispeech 292.367 samples
class AIHubDataset(torch.utils.data.Dataset): 
    def __init__(self, dataset_list, training_params, tokenizer_params, split, args):
        self.names = dataset_list 
        self.vocab_type = tokenizer_params["vocab_type"]
        self.vocab_size = str(tokenizer_params["vocab_size"])
        self.lm_mode = training_params.get("lm_mode", False)
        self.names = self.filter_lengths(training_params["train_audio_max_length"], training_params["train_label_max_length"], args.rank)

    def __getitem__(self, i):
        if self.lm_mode:
            return [torch.load(self.names[i] + "." + self.vocab_type + "_" + self.vocab_size)]
        else:
            return [torchaudio.load(self.names[i])[0], torch.load(self.names[i] + "." + self.vocab_type + "_" + self.vocab_size)]

    def __len__(self):

        return len(self.names)

    def filter_lengths(self, audio_max_length, label_max_length, rank=0):

        if audio_max_length is None or label_max_length is None:
            return self.names

        if rank == 0:
            print("AIHub dataset filtering")
            print("Audio maximum length : {} / Label sequence maximum length : {}".format(audio_max_length, label_max_length))
            self.names = tqdm(self.names)

        return [name for name in self.names if torch.load(name + "_len") <= audio_max_length and \
            torch.load(name+self.vocab_type + "_" + self.vocab_size + "_len") <= label_max_length]
            