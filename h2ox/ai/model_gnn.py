from typing import Dict, List, Optional

import numpy as np
import torch
from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator
from torch import nn

from h2ox.ai.gnn import GraphConvolution


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
class LayeredBayesianGraphLSTM(nn.Module):
    def __init__(
        self,
        sites: List[str],
        sites_edges: List[List[str]],
        flow_std: Dict[str, float],
        future_horizon: int,
        forecast_horizon: int,
        historic_period: int,
        historical_input_size: int,
        forecast_input_size: int,
        future_input_size: int,
        hidden_size: int,
        target_size: int,
        digraph: bool = False,
        diag: bool = True,
        num_layers: int = 1,
        dropout: float = 0.4,
        bayesian: bool = True,
        device: str = "cpu",
        lstm_params: Optional[dict] = None
        # include_current_timestep_in_horizon: bool = True,
    ):
        super().__init__()

        self.flow_std = flow_std
        self.sites = sites
        self.device = device
        self.hidden_size = hidden_size
        self.historic_period = historic_period
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

        self.encoders = nn.ModuleDict(
            {
                site: Encoder(layers_historic, device, bayesian, lstm_params).to(device)
                for site in sites
            }
        )
        # self.encoder_pre = nn.Linear(historical_input_size,hidden_size, bias=False)
        self.decoders_forecast = nn.ModuleDict(
            {
                site: Decoder(layers_forecast, device, bayesian, lstm_params).to(device)
                for site in sites
            }
        )
        # self.decoder_pre = nn.Linear(forecast_input_size, hidden_size, bias=False)
        self.decoders_future = nn.ModuleDict(
            {
                site: Decoder(layers_future, device, bayesian, lstm_params).to(device)
                for site in sites
            }
        )

        self.headers_in = nn.ModuleDict(
            {
                site: nn.Sequential(
                    nn.Linear(hidden_size + 1, hidden_size, bias=False),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                )
                for site in sites
            }
        )

        self.headers_out = nn.ModuleDict(
            {
                site: nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_size, 1, bias=False),
                )
                for site in sites
            }
        )

        self.gnn = GraphConvolution(hidden_size, hidden_size, bias=True)

        self.adj = np.zeros((len(sites), len(sites)), dtype=np.float32)

        for s1, s2 in sites_edges:
            self.adj[sites.index(s1), sites.index(s2)] = 1

        if diag:
            self.adj += np.diag(np.ones(len(sites)))

        if digraph:
            for s1, s2 in sites_edges:
                self.adj[sites.index(s2), sites.index(s1)] = -1

        self.adj = torch.from_numpy(self.adj).to(device)

    def absflow(self, flows, site):
        return flows * self.flow_std[site]

    def forward(self, data):
        batch_size = data[self.sites[0]]["x_d"].shape[0]
        # RUN HISTORY

        # feedforward the endocders
        hiddens, cells = {}, {}

        """
        for site in self.sites:
            _, hiddens[site], cells[site] = self.encoders[site](
                torch.cat([
                    data[site]['x_d'][:,0,:].unsqueeze(1),
                    data[site]['hist_level'][:,0,:].unsqueeze(1)
                ], dim=-1)
            )

        for r in range(1, self.historic_period):
            for site in self.sites:
                _, hiddens[site], cells[site] = self.encoders[site](
                    torch.cat([
                        data[site]['x_d'][:,0,:].unsqueeze(1),
                        data[site]['hist_level'][:,0,:].unsqueeze(1)
                    ], dim=-1),
                    hiddens[site],
                    cells[site]
                )
        """

        for site in self.sites:
            _, hiddens[site], cells[site] = self.encoders[site](data[site]["x_d"])

        # horizon size = forecast_horizon + future_horizon
        horizon = self.target_horizon
        outputs = torch.zeros(batch_size, horizon, len(self.sites)).to(self.device)

        site_levels = {site: data[site]["hist_level"][:, -1, :] for site in self.sites}

        # RUN FORECAST [0 -- 14]
        for t in range(0, self.forecast_horizon):

            output_t = torch.zeros(batch_size, len(self.sites), self.hidden_size).to(
                self.device
            )

            # predict each sitewise
            for ii_s, site in enumerate(self.sites):
                output, hiddens[site], cells[site] = self.decoders_forecast[site](
                    data[site]["x_f"][:, t, :].unsqueeze(1), hiddens[site], cells[site]
                )

                # print ('SHAPE')
                # print (output.shape)
                # print (site_levels[site].unsqueeze(-1).shape)
                # print (output_t[:,ii_s,:].shape)
                # print (torch.cat([
                #        output,
                #        site_levels[site].unsqueeze(-1)
                #    ], dim=-1).shape)

                # print (self.headers[site](
                #    torch.cat([
                #        output,
                #        site_levels[site].unsqueeze(-1)
                #    ], dim=-1)
                # ).shape)

                # put through first header
                output_t[:, ii_s, :] = self.headers_in[site](
                    torch.cat(
                        [output, site_levels[site].unsqueeze(-1)], dim=-1
                    ).squeeze()
                )

            # graph convolve
            graphed = self.gnn(output_t, self.adj)

            flows = torch.zeros(batch_size, len(self.sites), 1).to(self.device)
            for ii_s, site in enumerate(self.sites):
                flows[:, ii_s, :] = self.headers_out[site](graphed[:, ii_s, :])

            outputs[:, t, :] = flows.squeeze()

            # add target back to outputs
            site_levels = {
                site: (site_levels[site] + flows[:, ii_s] * self.flow_std[site])
                for ii_s, site in enumerate(self.sites)
            }

        # RUN FUTURE [14 -- 90]
        for t in range(self.forecast_horizon, self.target_horizon):

            output_t = torch.zeros(batch_size, len(self.sites), self.hidden_size).to(
                self.device
            )

            # predict each sitewise
            for ii_s, site in enumerate(self.sites):
                output, hiddens[site], cells[site] = self.decoders_future[site](
                    data[site]["x_ff"][:, t - self.forecast_horizon, :].unsqueeze(1),
                    hiddens[site],
                    cells[site],
                )

                # put through header
                output_t[:, ii_s, :] = self.headers_in[site](
                    torch.cat(
                        [output, site_levels[site].unsqueeze(-1)], dim=-1
                    ).squeeze()
                )

            # graph convolve
            graphed = self.gnn(output_t, self.adj)

            flows = torch.zeros(batch_size, len(self.sites), 1).to(self.device)
            for ii_s, site in enumerate(self.sites):
                flows[:, ii_s, :] = self.headers_out[site](graphed[:, ii_s, :])

            outputs[:, t, :] = flows.squeeze()

            # add target back to outputs
            site_levels = {
                site: (site_levels[site] + flows[:, ii_s] * self.flow_std[site])
                for ii_s, site in enumerate(self.sites)
            }

        return outputs


def initialise_gnn(
    item: dict,
    sites: List[str],
    sites_edges: List[List[str]],
    flow_std: Dict[str, float],
    device: str,
    diag: bool = True,
    digraph: bool = False,
    hidden_size: int = 6,
    num_layers: int = 1,
    dropout: float = 0.4,
    bayesian: bool = True,
    lstm_params: Optional[dict] = None,
) -> LayeredBayesianGraphLSTM:
    # initialise model shapes
    forecast_horizon = item[sites[0]]["x_f"].shape[0]
    future_horizon = item[sites[0]]["x_ff"].shape[0]

    historic_period = item[sites[0]]["x_d"].shape[0]
    historical_input_size = item[sites[0]]["x_d"].shape[1]
    forecast_input_size = item[sites[0]]["x_f"].shape[1]
    future_input_size = item[sites[0]]["x_ff"].shape[1]
    target_size = item[sites[0]]["y"].shape[1]

    if lstm_params is None:
        lstm_params = {}

    print("forecast_horizon", forecast_horizon)
    print("future_horizon", future_horizon)
    print("historic", historical_input_size)
    print("forecast", forecast_input_size)
    print("future", future_input_size)
    print("target", target_size)

    model = LayeredBayesianGraphLSTM(
        sites=sites,
        sites_edges=sites_edges,
        flow_std=flow_std,
        digraph=digraph,
        diag=diag,
        future_horizon=future_horizon,
        forecast_horizon=forecast_horizon,
        historic_period=historic_period,
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
