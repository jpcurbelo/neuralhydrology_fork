from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class CudaLSTM(BaseModel):
    """LSTM model class, which relies on PyTorch's CUDA LSTM class.

    This class implements the standard LSTM combined with a model head, as specified in the config. Depending on the
    embedding settings, static and/or dynamic features may or may not be fed through embedding networks before being
    concatenated and passed through the LSTM.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow.
    The `CudaLSTM` class only supports single-timescale predictions. Use `MTSLSTM` to train a model and get
    predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(CudaLSTM, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        
        # # Add Batch Normalization layer
        # self.batch_norm = nn.BatchNorm1d(self.embedding_net.output_size)
               
        # Check if list
        if isinstance(cfg.hidden_size, list):
            self.n_layers = len(cfg.hidden_size)
            self.n_units = cfg.hidden_size[0]
        elif isinstance(cfg.hidden_size, int):
            self.n_layers = 1
            self.n_units = cfg.hidden_size

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=self.n_units, num_layers=self.n_layers)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=self.n_units, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.n_units:2 * self.n_units] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)
        
        # # Apply batch normalization before LSTM
        # x_d = self.batch_norm(x_d.permute(1, 2, 0)).permute(2, 0, 1)

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)
        
        # reshape to [batch_size, seq, n_hiddens]
        lstm_output = lstm_output.transpose(0, 1)
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'lstm_output': lstm_output, 'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output)))

        return pred
