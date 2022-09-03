import os
import numpy as np
import torchaudio

import torch
import torch.nn as nn
from torch import Tensor

from modules.vocab import KoreanSpeechVocabulary
from modules.data import load_audio
from modules.model import DeepSpeech2
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, build_ctcdecoder


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)
    return torch.FloatTensor(feature).transpose(0, 1)


def single_infer(model, audio_path):
    device = 'cuda'
    feature = parse_audio(audio_path, del_silence=True)
    input_length = torch.LongTensor([len(feature)])
    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')
    decoder = build_ctcdecoder(labels=list(vocab.vocab_dict.keys()))
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()
    model.device = device
    y_hats = model(feature.unsqueeze(0).to(device), input_length)
    sentence = decoder.decode_beams(y_hats[0][0].cpu().detach().numpy())[0][0].replace("<eos>","").replace("<blank>","")

    return [sentence] 