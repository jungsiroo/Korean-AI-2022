import torch.nn as nn
from modules.conformer_lstm.encoder import ConformerEncoder
from modules.conformer_lstm.decoder import DecoderRNN
class ConformerLSTMModel(nn.Module):
    r"""
    Conformer encoder + LSTM decoder.
    Args:
        configs (DictConfig): configuraion set
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.
    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
    Returns:
        outputs (torch.FloatTensor): Result of model predictions.
    """
    def __init__(self, configs, vocab) -> None:
        super(ConformerLSTMModel, self).__init__()
        self.configs = configs
        self.vocab = vocab
        self.encoder = ConformerEncoder(
            num_classes=len(vocab),
            input_dim=self.configs.n_mels,
            encoder_dim=self.configs.encoder_dim,
            num_layers=self.configs.num_encoder_layers,
            num_attention_heads=self.configs.num_attention_heads,
            feed_forward_expansion_factor=self.configs.feed_forward_expansion_factor,
            conv_expansion_factor=self.configs.conv_expansion_factor,
            input_dropout_p=self.configs.input_dropout_p,
            feed_forward_dropout_p=self.configs.feed_forward_dropout_p,
            attention_dropout_p=self.configs.attention_dropout_p,
            conv_dropout_p=self.configs.conv_dropout_p,
            conv_kernel_size=self.configs.conv_kernel_size,
            joint_ctc_attention=True,
        )
        self.decoder = DecoderRNN(
            num_classes=len(vocab),
            max_length=256,
            hidden_state_dim=configs.encoder_dim,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
            num_layers=configs.num_decoder_layers,
            rnn_type=configs.rnn_type
        )
    
    def forward(self, inputs, input_lengths):
        _, encoder_outputs, _ = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(
            encoder_outputs=encoder_outputs, 
            teacher_forcing_ratio=0.0
        )
        return y_hats