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

# SentencePiece
import sentencepiece as spm

# Other
import sys
import glob
import re
import os
import pandas as pd

import nsml
from nsml import DATASET_PATH

def collate_fn_pad(batch):

    # Regular Mode
    if len(batch[0]) == 2:

        # Sorting sequences by lengths
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)

        # Pad data sequences
        data = [item[0].squeeze() for item in sorted_batch]
        data_lengths = torch.tensor([len(d) for d in data],dtype=torch.long) 
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

        # Pad labels
        target = [item[1] for item in sorted_batch]
        target_lengths = torch.tensor([t.size(0) for t in target],dtype=torch.long)
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)

        return data, target, data_lengths, target_lengths

    # LM Mode
    elif len(batch[0]) == 1:

        # Sort Batch
        sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
        sorted_batch = [item[0] for item in sorted_batch]

        # Create Labels
        x = torch.nn.utils.rnn.pad_sequence(sorted_batch, batch_first=True, padding_value=0)
        x_len = torch.tensor([t.size(0) for t in sorted_batch], dtype=torch.long)
        y = [torch.cat([item, item.new_zeros(1)]) for item in sorted_batch]
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)

        return x, x_len, y

    else:

        raise Exception("Batch Format Error")

def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic':
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        #########
        # There is no special token in train sentence 
        if ch == '#':
            new_sentence += '샾'
        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'
        #########

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)

def create_tokenizer(training_params, tokenizer_params):

    # AIHUB Dataset
    if training_params["training_dataset"] == "AIHub":

        # Corpus File Path
        corpus_path = f"{DATASET_PATH}/train/corpus.txt"
        data_df = pd.read_csv(f"{DATASET_PATH}/train/train_label")

        # Create Corpus File
        if not os.path.isfile(corpus_path):
            print("Create Corpus File")
            corpus_file = open(corpus_path, "w")
            for _, transcript in data_df.values:
                corpus_file.write(sentence_filter(transcript)+"\n")

        # Train Tokenizer
        print("Training Tokenizer")
        spm.SentencePieceTrainer.train(
            input=corpus_path, 
            model_prefix=f"{DATASET_PATH}/train/AIHub_bpe_256", 
            vocab_size=tokenizer_params["vocab_size"], 
            character_coverage=1.0, 
            model_type=tokenizer_params["vocab_type"], 
            bos_id=-1, eos_id=-1, unk_surface=""
        )
        print("Training Done")

def prepare_dataset(training_params, tokenizer_params, tokenizer):

    # LibriSpeech Dataset
    if training_params["training_dataset"] == "AIHub":

        # Read corpus
        print("Reading Corpus")
        label_paths = []
        sentences = []

        data_df = pd.read_csv(f"{DATASET_PATH}/train/train_label")

        for path, trans in data_df:
            label_paths.append(f"{DATASET_PATH}/train/{path}.{tokenizer_params['vocab_type']}_{str(tokenizer_params['vocab_size'])}")
            sentences.append(sentence_filter(trans))

        # for file_path in glob.glob(training_params["training_dataset_path"] + "*/*/*/*.txt"):
        #     for line in open(file_path, "r").readlines():
        #         label_paths.append(file_path.replace(file_path.split("/")[-1], "") + line.split()[0] + "." + tokenizer_params["vocab_type"] + "_" + str(tokenizer_params["vocab_size"]))
        #         sentences.append(line[len(line.split()[0]) + 1:-1].lower())

        # Save Labels and lengths
        print("Encoding sequences")
        for i, (sentence, label_path) in enumerate(zip(sentences, label_paths)):
            
            # Print
            sys.stdout.write("\r{}/{}".format(i, len(label_paths)))

            # Tokenize and Save label
            label = torch.LongTensor(tokenizer.encode(sentence))
            torch.save(label, label_path)

            # Save Audio length
            audio_length = torchaudio.load(label_path.split(".")[0])[0].size(1)
            torch.save(audio_length, label_path.split(".")[0] + ".pcm_len")

            # Save Label length
            label_length = label.size(0)
            torch.save(label_length, label_path + "_len")