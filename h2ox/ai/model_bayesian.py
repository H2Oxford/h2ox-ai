from typing import Dict, List, Optional

import numpy as np
import torch
from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self, layers: List[List[int]], device: str, bayesian: bool, lstm_params: dict
    ):
        super().__init__()

        self.bayesian = bayesian
        self.layers = layers

        if bayesian:

            lstms = {}
            for kk, (inp_size, hidden_size) in enumerate(layers):
                lstms[str(kk)] = BayesianLSTM(inp_size, hidden_size, **lstm_params).to(
                    device
                )

            self.lstms = nn.ModuleDict(lstms)

        else:
            self.lstm = nn.LSTM(
                input_size=layers[0][0],
                hidden_size=layers[-1][-1],
                batch_first=True,
                num_layers=len(layers),
            )
            self.hidden_size = layers[-1][-1]
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
        if self.bayesian:
            return self.forward_bayesian(data)
        else:
            return self.forward_nobayesian(data)

    def forward_nobayesian(self, data: Dict[str, torch.Tensor]):

        outputs, (hidden, cell) = self.lstm(data)

        return outputs, hidden, cell

    def forward_bayesian(self, data: Dict[str, torch.Tensor]):

        hiddens = []
        cells = []
        outp, (hidden, cell) = self.lstms[str(0)](data)

        hiddens.append(hidden)
        cells.append(cell)

        for kk in range(1, len(self.layers)):

            outp, (hidden, cell) = self.lstms[str(kk)](outp)
            hiddens.append(hidden)
            cells.append(cell)

        return outp, torch.stack(hiddens, dim=1), torch.stack(cells, dim=1)

    def forward_deterministic(self, data: Dict[str, torch.Tensor]):

        hiddens = []
        cells = []
        outp, (hidden, cell) = self.lstms[str(0)].forward_frozen(
            data, hidden_states=None
        )

        hiddens.append(hidden)
        cells.append(cell)

        for kk in range(1, len(self.layers)):

            outp, (hidden, cell) = self.lstms[str(kk)].forward_frozen(
                outp, hidden_states=None
            )
            hiddens.append(hidden)
            cells.append(cell)

        return outp, torch.stack(hiddens, dim=1), torch.stack(cells, dim=1)


class Decoder(nn.Module):
    def __init__(
        self, layers: List[List[int]], device: str, bayesian: bool, lstm_params: dict
    ):
        super().__init__()
        self.layers = layers

        self.bayesian = bayesian

        if bayesian:

            lstms = {}
            for kk, (inp_size, hidden_size) in enumerate(layers):
                print(inp_size, hidden_size)
                lstms[str(kk)] = BayesianLSTM(inp_size, hidden_size, **lstm_params).to(
                    device
                )

            self.lstms = nn.ModuleDict(lstms)

        else:

            self.lstm = nn.LSTM(
                input_size=layers[0][0],
                hidden_size=layers[-1][-1],
                batch_first=True,
                num_layers=len(layers),
            )
            self.hidden_size = layers[-1][-1]
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

    def forward(self, X: torch.Tensor, hidden, cell):
        if self.bayesian:
            return self.forward_bayesian(X, hidden, cell)
        else:
            return self.forward_nobayesian(X, hidden, cell)

    def forward_nobayesian(self, x_f: torch.Tensor, hidden, cell):
        # NOTE: data dict is already processed in parent model!
        output, (hidden, cell) = self.lstm(x_f, (hidden, cell))

        return output, hidden, cell

    def forward_bayesian(self, X: torch.Tensor, hidden, cell):

        new_hiddens = []
        new_cells = []
        outp, (new_hidden, new_cell) = self.lstms[str(0)](
            X, (hidden[:, 0, :], cell[:, 0, :])
        )

        new_hiddens.append(new_hidden)
        new_cells.append(new_cell)

        # probably need to unpack the hidden state

        for kk in range(1, len(self.layers)):

            outp, (new_hidden, new_cell) = self.lstms[str(kk)](
                outp, (hidden[:, kk, :], cell[:, kk, :])
            )
            new_hiddens.append(new_hidden)
            new_cells.append(new_cell)

        return outp, torch.stack(new_hiddens, dim=1), torch.stack(new_cells, dim=1)

    def forward_deterministic(self, X: torch.Tensor, hidden, cell):

        new_hiddens = []
        new_cells = []
        outp, (new_hidden, new_cell) = self.lstms[str(0)].forward_frozen(
            X, (hidden[:, 0, :], cell[:, 0, :])
        )

        new_hiddens.append(new_hidden)
        new_cells.append(new_cell)

        # probably need to unpack the hidden state

        for kk in range(1, len(self.layers)):

            outp, (new_hidden, new_cell) = self.lstms[str(kk)].forward_frozen(
                outp, (hidden[:, kk, :], cell[:, kk, :])
            )
            new_hiddens.append(new_hidden)
            new_cells.append(new_cell)

        return outp, torch.stack(new_hiddens, dim=1), torch.stack(new_cells, dim=1)


@variational_estimator
class LayeredBayesianLSTM(nn.Module):
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
        bayesian: bool = True,
        device: str = "cpu",
        lstm_params: Optional[dict] = None
        # include_current_timestep_in_horizon: bool = True,
    ):
        super().__init__()

        self.device = device
        self.forecast_horizon = forecast_horizon
        self.future_horizon = future_horizon
        # TODO: calculate this from the data too!
        self.target_horizon = forecast_horizon + future_horizon
        self.target_size = target_size

        if lstm_params is None:
            lstm_params = {}

        layers_historic = [[historical_input_size, hidden_size]] + [
            [hidden_size, hidden_size]
        ] * (num_layers - 1)
        layers_forecast = [[forecast_input_size, hidden_size]] + [
            [hidden_size, hidden_size]
        ] * (num_layers - 1)
        layers_future = [[future_input_size, hidden_size]] + [
            [hidden_size, hidden_size]
        ] * (num_layers - 1)

        print("layers historic", layers_historic)
        print("layers forecast", layers_forecast)
        print("layers future", layers_future)

        self.encoder = Encoder(layers_historic, device, bayesian, lstm_params).to(
            device
        )
        # self.encoder_pre = nn.Linear(historical_input_size,hidden_size, bias=False)
        self.decoder_forecast = Decoder(
            layers_forecast, device, bayesian, lstm_params
        ).to(device)
        # self.decoder_pre = nn.Linear(forecast_input_size, hidden_size, bias=False)
        self.decoder_future = Decoder(layers_future, device, bayesian, lstm_params).to(
            device
        )

        # construct header
        dropout = nn.Dropout(p=dropout)
        # fc_layer = nn.Linear(hidden_size, target_size)
        assert (
            hidden_size % target_size == 0
        ), "hidden size must be multiple of target size"
        # stride = kernel = hidden_size // target_size  # 36 // 6
        # conv1d = torch.nn.Conv1d(1, 1, kernel, stride=stride, bias=False)
        # conv1d.bias.data.fill_(0.)
        # conv1d.weight.data.fill_(0.01)
        head_linear = nn.Linear(hidden_size, target_size, bias=False)

        self.head = nn.Sequential(dropout, head_linear)

    def forward(self, data):
        batch_size = data["x_d"].shape[0]
        # RUN HISTORY
        # encoder_outputs == [batch_size, seq_length, hidden_size]
        encoder_outputs, hidden, cell = self.encoder(
            data["x_d"]
        )  # self.encoder(self.encoder_pre(data['x_d']))

        # horizon size = forecast_horizon + future_horizon
        horizon = self.target_horizon
        outputs = torch.zeros(batch_size, horizon, self.target_size).to(self.device)

        # RUN FORECAST [0 -- 14]
        for t in range(0, self.forecast_horizon):
            # x_f = self.decoder_pre(data["x_f"][:, t, :].unsqueeze(1))
            x_f = data["x_f"][:, t, :].unsqueeze(1)

            output, hidden, cell = self.decoder_forecast(x_f, hidden, cell)
            pred = self.head(output)

            # pass through the linear head (SAME AS FUTURE HEAD (?))

            outputs[:, t, :] = pred.squeeze()

        # RUN FUTURE [14 -- 90]
        for t in range(self.forecast_horizon, self.target_horizon):
            x_ff = data["x_ff"][:, t - self.forecast_horizon, :].unsqueeze(1)

            output, hidden, cell = self.decoder_future(x_ff, hidden, cell)

            # pass through the linear head (SAME AS FORECAST HEAD (?))
            pred = self.head(output)

            outputs[:, t, :] = pred.squeeze()

        return outputs

    def forward_deterministic(self, data):

        batch_size = data["x_d"].shape[0]

        encoder_outputs, hidden, cell = self.encoder.forward_deterministic(
            data["x_d"].to(self.device)
        )

        # horizon size = forecast_horizon + future_horizon
        horizon = self.target_horizon
        outputs = torch.zeros(batch_size, horizon, self.target_size).to(self.device)

        # RUN FORECAST [0 -- 14]
        for t in range(0, self.forecast_horizon):
            x_f = data["x_f"][:, t, :].unsqueeze(1)  # .to(self.device)

            output, hidden, cell = self.decoder_forecast.forward_deterministic(
                x_f, hidden, cell
            )

            pred = self.head(output)
            # pass through the linear head (SAME AS FUTURE HEAD (?))

            outputs[:, t, :] = pred.squeeze()

        # RUN FUTURE [14 -- 90]
        for t in range(self.forecast_horizon, self.target_horizon):
            x_ff = data["x_ff"][:, t - self.forecast_horizon, :].unsqueeze(
                1
            )  # .to(self.device)

            output, hidden, cell = self.decoder_future.forward_deterministic(
                x_ff, hidden, cell
            )

            # pass through the linear head (SAME AS FORECAST HEAD (?))
            pred = self.head(output)

            outputs[:, t, :] = pred.squeeze()

        return outputs


def initialise_bayesian(
    item: dict,
    device: str,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.4,
    bayesian: bool = True,
    lstm_params: Optional[dict] = None,
) -> LayeredBayesianLSTM:

    # initialise model shapes
    forecast_horizon = item["x_f"].shape[0]
    future_horizon = item["x_ff"].shape[0]

    historical_input_size = item["x_d"].shape[1]
    forecast_input_size = item["x_f"].shape[1]
    future_input_size = item["x_ff"].shape[1]
    target_size = item["y"].shape[1]

    if lstm_params is None:
        lstm_params = {}

    print("forecast_horizon", forecast_horizon)
    print("future_horizon", future_horizon)
    print("historic", historical_input_size)
    print("forecast", forecast_input_size)
    print("future", future_input_size)
    print("target", target_size)

    model = LayeredBayesianLSTM(
        future_horizon=future_horizon,
        forecast_horizon=forecast_horizon,
        historical_input_size=historical_input_size,
        forecast_input_size=forecast_input_size,
        future_input_size=future_input_size,
        hidden_size=hidden_size,
        target_size=target_size,
        num_layers=num_layers,
        dropout=dropout,
        bayesian=bayesian,
        device=device,
        lstm_params=lstm_params,
    )

    return model


if __name__ == "__main__":

    test_model = LayeredBayesianLSTM(
        future_horizon=76,
        forecast_horizon=14,
        historical_input_size=2 * 6 + 2,
        forecast_input_size=2 * 6 + 2,
        future_input_size=2,
        hidden_size=36,
        target_size=6,
        num_layers=1,
        dropout=0.4,
        device="cuda:0",
    ).to("cuda:0")

    for param in test_model.parameters():
        print(param)
        # print ([p for p in layer.parameters()])

    data = {
        "x_d": torch.rand(100, 30, 2 * 6 + 2).to("cuda:0"),
        "x_f": torch.rand(100, 14, 2 * 6 + 2).to("cuda:0"),
        "x_ff": torch.rand(100, 76, 2).to("cuda:0"),
    }

    result = test_model(data)
    print("results", result.shape)
