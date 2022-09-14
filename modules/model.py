from typing import Tuple
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from astropy.modeling import ParameterError

from modules.vocab import Vocabulary

from modules.models.conformer import Conformer
from modules.models.convolution import *
from modules.vocabs import Vocabulary
from modules.models.las import EncoderRNN
from modules.decode.ensemble import (
    BasicEnsemble,
    WeightedEnsemble,
)
from modules.models import (
    ListenAttendSpell,
    DeepSpeech2,
    SpeechTransformer,
    Jasper,
    RNNTransducer,
)

def build_model(
        config,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:
    """ Various model dispatcher function. """
    if config.transform_method.lower() == 'spect':
        if config.feature_extract_by == 'kaldi':
            input_size = 257
        else:
            input_size = (config.frame_length << 3) + 1
    else:
        input_size = config.n_mels

    if config.architecture.lower() == 'las':
        model = build_las(input_size, config, vocab, device)

    elif config.architecture.lower() == 'transformer':
        model = build_transformer(
            num_classes=len(vocab),
            input_dim=input_size,
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_heads=config.num_heads,
            pad_id=vocab.pad_id,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            max_length=config.max_len,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dropout_p=config.dropout,
            device=device,
            joint_ctc_attention=config.joint_ctc_attention,
            extractor=config.extractor,
        )

    elif config.architecture.lower() == 'deepspeech2':
        model = build_deepspeech2(
            input_size=input_size,
            num_classes=len(vocab),
            rnn_type=config.rnn_type,
            num_rnn_layers=config.num_encoder_layers,
            rnn_hidden_dim=config.hidden_dim,
            dropout_p=config.dropout,
            bidirectional=config.use_bidirectional,
            activation=config.activation,
            device=device,
        )

    elif config.architecture.lower() == 'jasper':
        model = build_jasper(
            num_classes=len(vocab),
            version=config.version,
            device=device,
        )

    elif config.architecture.lower() == 'conformer':
        model = build_conformer(
            num_classes=len(vocab),
            input_size=input_size,
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            decoder_rnn_type=config.decoder_rnn_type,
            num_attention_heads=config.num_attention_heads,
            feed_forward_expansion_factor=config.feed_forward_expansion_factor,
            conv_expansion_factor=config.conv_expansion_factor,
            input_dropout_p=config.input_dropout_p,
            feed_forward_dropout_p=config.feed_forward_dropout_p,
            attention_dropout_p=config.attention_dropout_p,
            conv_dropout_p=config.conv_dropout_p,
            decoder_dropout_p=config.decoder_dropout_p,
            conv_kernel_size=config.conv_kernel_size,
            half_step_residual=config.half_step_residual,
            device=device,
            decoder=config.decoder,
        )

    elif config.architecture.lower() == 'rnnt':
        model = build_rnnt(
            num_classes=len(vocab),
            input_dim=input_size,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            encoder_hidden_state_dim=config.encoder_hidden_state_dim,
            decoder_hidden_state_dim=config.decoder_hidden_state_dim,
            output_dim=config.output_dim,
            rnn_type=config.rnn_type,
            bidirectional=config.bidirectional,
            encoder_dropout_p=config.encoder_dropout_p,
            decoder_dropout_p=config.decoder_dropout_p,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
        )

    else:
        raise ValueError('Unsupported model: {0}'.format(config.architecture))

    print(model)

    return model

# def build_model(
#         config,
#         vocab: Vocabulary,
#         device: torch.device,
# ) -> nn.DataParallel:

#     input_size = config.n_mels

#     model = build_deepspeech2(
#         input_size=input_size,
#         num_classes=len(vocab),
#         rnn_type=config.rnn_type,
#         num_rnn_layers=config.num_encoder_layers,
#         rnn_hidden_dim=config.hidden_dim,
#         dropout_p=config.dropout,
#         bidirectional=config.use_bidirectional,
#         activation=config.activation,
#         device=device,
#     )

#     return model

class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class BNReluRNN(nn.Module):
    """
    Recurrent neural network with batch normalization layer & ReLU activation function.

    Args:
        input_size (int): size of input
        hidden_state_dim (int): the number of features in the hidden state `h`
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        dropout_p (float, optional): dropout probability (default: 0.1)

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs
        - **outputs**: Tensor produced by the BNReluRNN module
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_size: int,  # size of input
            hidden_state_dim: int = 512,  # dimension of RNN`s hidden state
            rnn_type: str = 'gru',  # type of RNN cell
            bidirectional: bool = True,  # if True, becomes a bidirectional rnn
            dropout_p: float = 0.1,  # dropout probability
    ):
        super(BNReluRNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.batch_norm = nn.BatchNorm1d(input_size)
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        total_length = inputs.size(0)

        inputs = F.relu(self.batch_norm(inputs.transpose(1, 2)))
        inputs = inputs.transpose(1, 2)

        outputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu())
        outputs, hidden_states = self.rnn(outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, total_length=total_length)

        return outputs


class MaskCNN(nn.Module):
    """
    Masking Convolutional Neural Network

    Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch sizes change during inference.
    Input needs to be in the shape of (batch_size, channel, hidden_dim, seq_len)

    Refer to https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Copyright (c) 2017 Sean Naren
    MIT License

    Args:
        sequential (torch.nn): sequential list of convolution layer

    Inputs: inputs, seq_lengths
        - **inputs** (torch.FloatTensor): The input of size BxCxHxT
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch

    Returns: output, seq_lengths
        - **output**: Masked output from the sequential
        - **seq_lengths**: Sequence length of output from the sequential
    """
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskCNN, self).__init__()
        self.sequential = sequential

    def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            seq_lengths = self._get_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        """
        Calculate convolutional neural network receptive formula

        Args:
            module (torch.nn.Module): module of CNN
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch

        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()


class Conv2dExtractor(nn.Module):
    """
    Provides inteface of convolutional extractor.

    Note:
        Do not use this class directly, use one of the sub classes.
        Define the 'self.conv' class variable.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """
    supported_activations = {
        'hardtanh': nn.Hardtanh(0, 20, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'gelu': nn.GELU(),
        'swish': Swish(),
    }

    def __init__(self, input_dim: int, activation: str = 'hardtanh') -> None:
        super(Conv2dExtractor, self).__init__()
        self.input_dim = input_dim
        self.activation = Conv2dExtractor.supported_activations[activation]
        self.conv = None

    def get_output_lengths(self, seq_lengths: Tensor):
        assert self.conv is not None, "self.conv should be defined"

        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
                seq_lengths = numerator.float() / float(module.stride[1])
                seq_lengths = seq_lengths.int() + 1

            elif isinstance(module, nn.MaxPool2d):
                seq_lengths >>= 1

        return seq_lengths.int()

    def get_output_dim(self):
        if isinstance(self, VGGExtractor):
            output_dim = (self.input_dim - 1) << 5 if self.input_dim % 2 else self.input_dim << 5

        elif isinstance(self, DeepSpeech2Extractor):
            output_dim = int(math.floor(self.input_dim + 2 * 20 - 41) / 2 + 1)
            output_dim = int(math.floor(output_dim + 2 * 10 - 21) / 2 + 1)
            output_dim <<= 5

        elif isinstance(self, Conv2dSubsampling):
            factor = ((self.input_dim - 1) // 2 - 1) // 2
            output_dim = self.out_channels * factor

        else:
            raise ValueError(f"Unsupported Extractor : {self.extractor}")

        return output_dim

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        inputs: torch.FloatTensor (batch, time, dimension)
        input_lengths: torch.IntTensor (batch)
        """
        outputs, output_lengths = self.conv(inputs.unsqueeze(1).transpose(2, 3), input_lengths)

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)

        return outputs, output_lengths


class VGGExtractor(Conv2dExtractor):
    """
    VGG extractor for automatic speech recognition described in
    "Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM" paper
    - https://arxiv.org/pdf/1706.02737.pdf

    Args:
        input_dim (int): Dimension of input vector
        in_channels (int): Number of channels in the input image
        out_channels (int or tuple): Number of channels produced by the convolution
        activation (str): Activation function

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """
    def __init__(
            self,
            input_dim: int,
            in_channels: int = 1,
            out_channels: int or tuple = (64, 128),
            activation: str = 'hardtanh',
    ):
        super(VGGExtractor, self).__init__(input_dim=input_dim, activation=activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                self.activation,
                nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                self.activation,
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                self.activation,
                nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                self.activation,
                nn.MaxPool2d(2, stride=2),
            )
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        return super().forward(inputs, input_lengths)

class DeepSpeech2Extractor(Conv2dExtractor):
    """
    DeepSpeech2 extractor for automatic speech recognition described in
    "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin" paper
    - https://arxiv.org/abs/1512.02595

    Args:
        input_dim (int): Dimension of input vector
        in_channels (int): Number of channels in the input vector
        out_channels (int): Number of channels produced by the convolution
        activation (str): Activation function

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs, output_lengths
        - **outputs**: Tensor produced by the convolution
        - **output_lengths**: Tensor containing sequence lengths produced by the convolution
    """
    def __init__(
            self,
            input_dim: int,
            in_channels: int = 1,
            out_channels: int = 32,
            activation: str = 'hardtanh',
    ) -> None:
        super(DeepSpeech2Extractor, self).__init__(input_dim=input_dim, activation=activation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,
            )
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        return super().forward(inputs, input_lengths)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
        raise NotImplementedError


class EncoderModel(BaseModel):
    """ Super class of KoSpeech's Encoder only Models """
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.decoder = None

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  ctc training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor):

            * predicted_log_prob (torch.FloatTensor)s: Log probability of model predictions.
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, predicted_log_probs: Tensor) -> Tensor:
        """
        Decode encoder_outputs.

        Args:
            predicted_log_probs (torch.FloatTensor):Log probability of model predictions. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        return predicted_log_probs.max(-1)[1]

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Recognize input speech.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        predicted_log_probs, _ = self.forward(inputs, input_lengths)
        if self.decoder is not None:
            return self.decoder.decode(predicted_log_probs)
        return self.decode(predicted_log_probs)


# class DeepSpeech2(EncoderModel):
#     """
#     Deep Speech2 model with configurable encoder and decoder.
#     Paper: https://arxiv.org/abs/1512.02595

#     Args:
#         input_dim (int): dimension of input vector
#         num_classes (int): number of classfication
#         rnn_type (str, optional): type of RNN cell (default: gru)
#         num_rnn_layers (int, optional): number of recurrent layers (default: 5)
#         rnn_hidden_dim (int): the number of features in the hidden state `h`
#         dropout_p (float, optional): dropout probability (default: 0.1)
#         bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
#         activation (str): type of activation function (default: hardtanh)
#         device (torch.device): device - 'cuda' or 'cpu'

#     Inputs: inputs, input_lengths
#         - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
#         - **input_lengths**: list of sequence lengths

#     Returns: output
#         - **output**: tensor containing the encoded features of the input sequence
#     """
#     def __init__(
#             self,
#             input_dim: int,
#             num_classes: int,
#             rnn_type='gru',
#             num_rnn_layers: int = 5,
#             rnn_hidden_dim: int = 512,
#             dropout_p: float = 0.1,
#             bidirectional: bool = True,
#             activation: str = 'hardtanh',
#             device: torch.device = 'cuda',
#     ):
#         super(DeepSpeech2, self).__init__()
#         self.device = device
#         self.conv = DeepSpeech2Extractor(input_dim, activation=activation)
#         self.rnn_layers = nn.ModuleList()
#         rnn_output_size = rnn_hidden_dim << 1 if bidirectional else rnn_hidden_dim

#         for idx in range(num_rnn_layers):
#             self.rnn_layers.append(
#                 BNReluRNN(
#                     input_size=self.conv.get_output_dim() if idx == 0 else rnn_output_size,
#                     hidden_state_dim=rnn_hidden_dim,
#                     rnn_type=rnn_type,
#                     bidirectional=bidirectional,
#                     dropout_p=dropout_p,
#                 )
#             )

#         self.fc = nn.Sequential(
#             nn.LayerNorm(rnn_output_size),
#             Linear(rnn_output_size, num_classes, bias=False),
#         )

#     def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
#         """
#         Forward propagate a `inputs` for  ctc training.

#         Args:
#             inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
#                 `FloatTensor` of size ``(batch, seq_length, dimension)``.
#             input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

#         Returns:
#             (Tensor, Tensor):

#             * predicted_log_prob (torch.FloatTensor)s: Log probability of model predictions.
#             * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
#         """
#         outputs, output_lengths = self.conv(inputs, input_lengths)
#         outputs = outputs.permute(1, 0, 2).contiguous()

#         for rnn_layer in self.rnn_layers:
#             outputs = rnn_layer(outputs, output_lengths)

#         outputs = self.fc(outputs.transpose(0, 1)).log_softmax(dim=-1)

#         return outputs, output_lengths


def build_deepspeech2(
        input_size: int,
        num_classes: int,
        rnn_type: str,
        num_rnn_layers: int,
        rnn_hidden_dim: int,
        dropout_p: float,
        bidirectional: bool,
        activation: str,
        device: torch.device,
) -> nn.DataParallel:
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    if rnn_hidden_dim < 0:
        raise ParameterError("hidden_dim should be greater than 0")
    if num_rnn_layers < 0:
        raise ParameterError("num_layers should be greater than 0")

    return nn.DataParallel(DeepSpeech2(
        input_dim=input_size,
        num_classes=num_classes,
        rnn_type=rnn_type,
        num_rnn_layers=num_rnn_layers,
        rnn_hidden_dim=rnn_hidden_dim,
        dropout_p=dropout_p,
        bidirectional=bidirectional,
        activation=activation,
        device=device,
    )).to(device)

def build_rnnt(
        num_classes: int,
        input_dim: int,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        encoder_hidden_state_dim: int = 320,
        decoder_hidden_state_dim: int = 512,
        output_dim: int = 512,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        encoder_dropout_p: float = 0.2,
        decoder_dropout_p: float = 0.2,
        sos_id: int = 1,
        eos_id: int = 2,
) -> nn.DataParallel:
    return nn.DataParallel(RNNTransducer(
        num_classes=num_classes,
        input_dim=input_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        encoder_hidden_state_dim=encoder_hidden_state_dim,
        decoder_hidden_state_dim=decoder_hidden_state_dim,
        output_dim=output_dim,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        encoder_dropout_p=encoder_dropout_p,
        decoder_dropout_p=decoder_dropout_p,
        sos_id=sos_id,
        eos_id=eos_id,
    ))


def build_conformer(
        num_classes: int,
        input_size: int,
        encoder_dim: int,
        decoder_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_rnn_type: str,
        num_attention_heads: int,
        feed_forward_expansion_factor: int,
        conv_expansion_factor: int,
        input_dropout_p: float,
        feed_forward_dropout_p: float,
        attention_dropout_p: float,
        conv_dropout_p: float,
        decoder_dropout_p: float,
        conv_kernel_size: int,
        half_step_residual: bool,
        device: torch.device,
        decoder: str,
) -> nn.DataParallel:
    if input_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if feed_forward_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if attention_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if conv_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    assert conv_expansion_factor == 2, "currently, conformer conv expansion factor only supports 2"

    return nn.DataParallel(Conformer(
        num_classes=num_classes,
        input_dim=input_size,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        decoder_rnn_type=decoder_rnn_type,
        num_attention_heads=num_attention_heads,
        feed_forward_expansion_factor=feed_forward_expansion_factor,
        conv_expansion_factor=conv_expansion_factor,
        input_dropout_p=input_dropout_p,
        feed_forward_dropout_p=feed_forward_dropout_p,
        attention_dropout_p=attention_dropout_p,
        conv_dropout_p=conv_dropout_p,
        decoder_dropout_p=decoder_dropout_p,
        conv_kernel_size=conv_kernel_size,
        half_step_residual=half_step_residual,
        device=device,
        decoder=decoder,
    )).to(device)


def build_deepspeech2(
        input_size: int,
        num_classes: int,
        rnn_type: str,
        num_rnn_layers: int,
        rnn_hidden_dim: int,
        dropout_p: float,
        bidirectional: bool,
        activation: str,
        device: torch.device,
) -> nn.DataParallel:
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    if rnn_hidden_dim < 0:
        raise ParameterError("hidden_dim should be greater than 0")
    if num_rnn_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    if rnn_type.lower() not in EncoderRNN.supported_rnns.keys():
        raise ParameterError("Unsupported RNN Cell: {0}".format(rnn_type))

    return nn.DataParallel(DeepSpeech2(
        input_dim=input_size,
        num_classes=num_classes,
        rnn_type=rnn_type,
        num_rnn_layers=num_rnn_layers,
        rnn_hidden_dim=rnn_hidden_dim,
        dropout_p=dropout_p,
        bidirectional=bidirectional,
        activation=activation,
        device=device,
    )).to(device)


def build_transformer(
        num_classes: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        input_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        extractor: str,
        dropout_p: float,
        device: torch.device,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        joint_ctc_attention: bool = False,
        max_length: int = 400,
) -> nn.DataParallel:
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_dim < 0:
        raise ParameterError("input_size should be greater than 0")
    if num_encoder_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    if num_decoder_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    return nn.DataParallel(SpeechTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        extractor=extractor,
        d_model=d_model,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        encoder_dropout_p=dropout_p,
        decoder_dropout_p=dropout_p,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        max_length=max_length,
        joint_ctc_attention=joint_ctc_attention,
    )).to(device)


def build_las(
        input_size: int,
        config,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:
    model = ListenAttendSpell(
        input_dim=input_size,
        num_classes=len(vocab),
        encoder_hidden_state_dim=config.hidden_dim,
        decoder_hidden_state_dim=config.hidden_dim << (1 if config.use_bidirectional else 0),
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        bidirectional=config.use_bidirectional,
        extractor=config.extractor,
        activation=config.activation,
        rnn_type=config.rnn_type,
        max_length=config.max_len,
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
        attn_mechanism=config.attn_mechanism,
        num_heads=config.num_heads,
        encoder_dropout_p=config.dropout,
        decoder_dropout_p=config.dropout,
        joint_ctc_attention=config.joint_ctc_attention,
    )
    model.flatten_parameters()

    return nn.DataParallel(model).to(device)


def build_jasper(
    version: str,
    num_classes: int,
    device: torch.device,
) -> nn.DataParallel:
    assert version.lower() in ["10x5", "5x3"], "Unsupported Version: {}".format(version)
    return nn.DataParallel(Jasper(
        num_classes=num_classes,
        version=version,
        device=device,
    ))


def load_test_model(config, device: torch.device):
    model = torch.load(config.model_path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model.module.decoder.device = device
        model.module.encoder.device = device

    else:
        model.encoder.device = device
        model.decoder.device = device

    return model


def load_language_model(path: str, device: torch.device):
    model = torch.load(path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(model, nn.DataParallel):
        model = model.module

    model.device = device

    return model


def build_ensemble(model_paths: list, method: str, device: torch.device):
    models = list()

    for model_path in model_paths:
        models.append(torch.load(model_path, map_location=lambda storage, loc: storage))

    if method == 'basic':
        ensemble = BasicEnsemble(models).to(device)
    elif method == 'weight':
        ensemble = WeightedEnsemble(models).to(device)
    else:
        raise ValueError("Unsupported ensemble method : {0}".format(method))

    return ensemble