"""[summary]
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194889
https://arxiv.org/abs/2004.13408
"""
from typing import Dict, Optional
from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


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
        # self.target_horizon = forecast_horizon + future_horizon + 1 if self.include_current_timestep_in_horizon else forecast_horizon + future_horizon

        self.dropout = nn.Dropout(p=dropout)
        fc_layer = nn.Linear(hidden_size, 1)
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
        outputs = torch.zeros(batch_size, horizon if horizon > 0 else 1).to(self.device)

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

            # pass through the linear head (SAME AS FUTURE HEAD (?))
            # t_adj = t + 1 if self.include_current_timestep_in_horizon else t
            t_adj = t
            outputs[:, t_adj] = torch.squeeze(
                self.head(self.dropout(torch.squeeze(output)))
            )

        # RUN FUTURE [14 -- 90]
        for t in range(self.forecast_horizon, self.target_horizon):
            x_ff = data["x_ff"][:, t - self.forecast_horizon, :].unsqueeze(1)

            output, hidden, cell = self.decoder_future(x_ff, hidden, cell)

            # pass through the linear head (SAME AS FORECAST HEAD (?))
            # t_adj = t + 1 if self.include_current_timestep_in_horizon else t
            t_adj = t
            outputs[:, t_adj] = torch.squeeze(
                self.head(self.dropout(torch.squeeze(output)))
            )

        return outputs


def initialise_model(
    dd: Dataset, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.4
) -> S2S2SModel:
    # initialise model shapes
    forecast_horizon = dd.forecast_horizon
    future_horizon = dd.future_horizon

    historical_input_size = dd.historical_input_size
    forecast_input_size = dd.forecast_input_size
    future_input_size = dd.future_input_size

    model = S2S2SModel(
        forecast_horizon=forecast_horizon,
        future_horizon=future_horizon,
        historical_input_size=historical_input_size,
        forecast_input_size=forecast_input_size,
        future_input_size=future_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    return model


if __name__ == "__main__":
    # load data
    # load dataset
    # load dataloader
    # initialise model
    # run model forward
    from h2ox.ai.dataset import FcastDataset
    from h2ox.ai.scripts.utils import load_zscore_data
    from pathlib import Path
    from torch.utils.data import DataLoader

    # parameters for the yaml file
    ENCODE_DOY = True
    SEQ_LEN = 60
    FUTURE_HORIZON = 76
    SITE = "kabini"
    TARGET_VAR = "volume_bcm"
    HISTORY_VARIABLES = ["tp", "t2m"]
    FORECAST_VARIABLES = ["tp", "t2m"]
    FUTURE_VARIABLES = []
    BATCH_SIZE = 32
    TRAIN_END_DATE = "2011-01-01"
    TRAIN_START_DATE = "2010-01-01"
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    DROPOUT = 0.4
    NUM_WORKERS = 1
    N_EPOCHS = 10
    RANDOM_VAL_SPLIT = True

    # load data
    data_dir = Path(Path.cwd() / "data")
    target, history, forecast = load_zscore_data(data_dir)

    # get train data
    train_target = target.sel(time=slice(TRAIN_START_DATE, TRAIN_END_DATE))
    train_history = history.sel(time=slice(TRAIN_START_DATE, TRAIN_END_DATE))
    train_forecast = forecast.sel(
        initialisation_time=slice(TRAIN_START_DATE, TRAIN_END_DATE)
    )

    # # select site
    y = train_target.sel(location=[SITE])
    x_d = train_history.sel(location=[SITE])
    x_f = train_forecast.sel(location=[SITE])

    # load dataset
    dd = FcastDataset(
        target=y,  # target,
        history=x_d,  # history,
        forecast=None,  # forecast,
        encode_doy=ENCODE_DOY,
        historical_seq_len=SEQ_LEN,
        future_horizon=FUTURE_HORIZON,
        target_var=TARGET_VAR,
        history_variables=HISTORY_VARIABLES,
        forecast_variables=FORECAST_VARIABLES,
    )
    dl = DataLoader(dd, batch_size=BATCH_SIZE, shuffle=False)

    # initialise model shapes
    # initialise model shapes
    forecast_horizon = dd.forecast_horizon
    future_horizon = dd.future_horizon

    historical_input_size = dd.historical_input_size
    forecast_input_size = dd.forecast_input_size
    future_input_size = dd.future_input_size

    model = S2S2SModel(
        forecast_horizon=forecast_horizon,
        future_horizon=future_horizon,
        historical_input_size=historical_input_size,
        forecast_input_size=forecast_input_size,
        future_input_size=future_input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    data_example = dl.__iter__().__next__()
    yhat = model(data_example)
    # assert yhat.shape == (
    #     BATCH_SIZE,
    #     total_horizon,
    # ), f"Expected {(BATCH_SIZE, total_horizon)} Got: {yhat.shape}"

    # train -- validation split (for testing hyperparameters)
    train_size = int(0.8 * len(dd))
    validation_size = len(dd) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dd, [train_size, validation_size]
    )
