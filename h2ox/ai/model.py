"""[summary]
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194889
https://arxiv.org/abs/2004.13408
"""
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self._reset_parameters()
        self.initialize_weights()

    def _reset_parameters(self, initial_forget_bias: Optional[float] = 0.3):
        """Special initialization of certain model weights."""
        if initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[
                self.hidden_size : 2 * self.hidden_size
            ] = initial_forget_bias

    def initialize_weights(self):
        # We are initializing the weights here with Xavier initialisation
        #  (by multiplying with 1/sqrt(n))
        # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        sqrt_k = np.sqrt(1 / self.hidden_size)
        for parameters in self.lstm.parameters():
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)

    def forward(self, data: Dict[str, torch.Tensor]):
        #  process data dict
        x_d = data["x_d"]

        outputs, (hidden, cell) = self.lstm(x_d)

        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self._reset_parameters()
        self.initialize_weights()

    def _reset_parameters(self, initial_forget_bias: Optional[float] = 0.3):
        """Special initialization of certain model weights."""
        if initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[
                self.hidden_size : 2 * self.hidden_size
            ] = initial_forget_bias

    def initialize_weights(self):
        # We are initializing the weights here with Xavier initialisation
        #  (by multiplying with 1/sqrt(n))
        # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        sqrt_k = np.sqrt(1 / self.hidden_size)
        for parameters in self.lstm.parameters():
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)

    def forward(self, x_f: torch.Tensor, hidden, cell):
        # NOTE: data dict is already processed in parent model!
        output, (hidden, cell) = self.lstm(x_f, (hidden, cell))

        return output, hidden, cell


class S2S2SModel(nn.Module):
    def __init__(
        self,
        future_horizon: int,
        forecast_horizon: int,
        historical_input_size: int,
        forecast_input_size: int,
        future_input_size: int,
        hidden_size: int,
        target_size: int,
        num_layers: int = 1,
        dropout: float = 0.4,
        device: str = "cpu",
        # include_current_timestep_in_horizon: bool = True,
    ):
        super().__init__()
        # self.include_current_timestep_in_horizon = include_current_timestep_in_horizon
        # NOTE: forecast_input_size and historical_input_size
        # currently have static_size included/added on
        self.encoder = Encoder(
            input_size=historical_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.decoder_forecast = Decoder(
            input_size=forecast_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.decoder_future = Decoder(
            input_size=future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.device = device
        self.forecast_horizon = forecast_horizon
        self.future_horizon = future_horizon
        # TODO: calculate this from the data too!
        self.target_horizon = forecast_horizon + future_horizon
        self.target_size = target_size
        # self.target_horizon = forecast_horizon + future_horizon + 1 if self.include_current_timestep_in_horizon else forecast_horizon + future_horizon

        self.dropout = nn.Dropout(p=dropout)
        fc_layer = nn.Linear(hidden_size, target_size)
        self.head = nn.Sequential(*[fc_layer])

    def forward(self, data):
        batch_size = data["x_d"].shape[0]
        # RUN HISTORY
        # encoder_outputs == [batch_size, seq_length, hidden_size]
        encoder_outputs, hidden, cell = self.encoder(data)

        # forecast - passing hidden and cell from previous timestep forwards
        #  the zero-time forecast just using the output of the encoder.
        # horizon size = forecast_horizon + future_horizon
        horizon = self.target_horizon
        outputs = torch.zeros(batch_size, horizon, self.target_size).to(self.device)

        # RUN PREDICTION OF NOW (initialisation_time)
        # pass through the linear head (SAME AS FUTURE HEAD (?))
        # if self.include_current_timestep_in_horizon:
        #     outputs[:, 0] = torch.squeeze(
        #         self.head(self.dropout(torch.squeeze(encoder_outputs[:, -1, :])))
        #     )

        # RUN FORECAST [0 -- 14]
        for t in range(0, self.forecast_horizon):
            x_f = data["x_f"][:, t, :].unsqueeze(1)

            output, hidden, cell = self.decoder_forecast(x_f, hidden, cell)
            # print ('output_shape',output.shape)
            pred = self.head(self.dropout(torch.squeeze(output)))
            # print ('pred shape',pred.shape)

            # pass through the linear head (SAME AS FUTURE HEAD (?))

            outputs[:, t, :] = pred

        # RUN FUTURE [14 -- 90]
        for t in range(self.forecast_horizon, self.target_horizon):
            x_ff = data["x_ff"][:, t - self.forecast_horizon, :].unsqueeze(1)

            output, hidden, cell = self.decoder_future(x_ff, hidden, cell)

            # pass through the linear head (SAME AS FORECAST HEAD (?))
            pred = self.head(self.dropout(torch.squeeze(output)))

            outputs[:, t, :] = pred

        return outputs


def initialise_model(
    item: dict, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.4
) -> S2S2SModel:
    # initialise model shapes
    forecast_horizon = item["x_f"].shape[0]
    future_horizon = item["x_ff"].shape[0]

    historical_input_size = item["x_d"].shape[1]
    forecast_input_size = item["x_f"].shape[1]
    future_input_size = item["x_ff"].shape[1]
    target_size = item["y"].shape[1]

    model = S2S2SModel(
        forecast_horizon=forecast_horizon,
        future_horizon=future_horizon,
        historical_input_size=historical_input_size,
        forecast_input_size=forecast_input_size,
        future_input_size=future_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        target_size=target_size,
        dropout=dropout,
    )

    return model
