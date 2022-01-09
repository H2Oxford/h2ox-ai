import logging, os, pickle, json

logging.basicConfig(level=logging.INFO)
from datetime import timedelta
import pandas as pd
from datetime import timedelta, datetime
from torchsummary import summary
from sklearn.metrics import r2_score

from tqdm import tqdm

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from src.w2w.dataloader_dnn import w2w_dnn_loader
from src.w2w.dataloader_lstm import w2w_lstm_loader
from matplotlib.collections import LineCollection


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


class S2S2SEncoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.rnn = nn.LSTM(
            inp_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )

    def forward(self, X):
        outputs, (hidden, cell) = self.rnn(X)

        return hidden, cell


class S2S2SDecoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.rnn = nn.LSTM(
            inp_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )

    def forward(self, X, hidden, cell):

        output, (hidden, cell) = self.rnn(X, (hidden, cell))

        return output, hidden, cell


class Seq2Seq2Seq(nn.Module):
    def __init__(self, lead_dim, horizon_dim, hidden_dim, n_layers, dropout, device):
        super().__init__()
        self.encoder = S2S2SEncoder(
            inp_dim=4, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout
        )
        self.decoder_1 = S2S2SDecoder(
            inp_dim=3, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout
        )
        self.decoder_2 = S2S2SDecoder(
            inp_dim=1, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout
        )
        self.device = device
        self.horizon_dim = horizon_dim

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, X):

        batch_size = X["historic"].shape[0]

        hidden, cell = self.encoder(X["historic"])

        outputs = torch.zeros(batch_size, self.horizon_dim).to(self.device)

        for t in range(14):

            output, hidden, cell = self.decoder_1(
                X["future_1"][:, t, :].unsqueeze(1), hidden, cell
            )

            output = self.fc_out(torch.squeeze(output))

            outputs[:, t] = torch.squeeze(output)

        for t in range(14, self.horizon_dim):
            output, hidden, cell = self.decoder_2(
                X["future_2"][:, t - 14, :].unsqueeze(1), hidden, cell
            )

            outputs[:, t] = torch.squeeze(self.fc_out(torch.squeeze(output)))

        return outputs


def train(
    model, trn_loader, val_loader, optimizer, PARAMS, device, savepath, verbose=True
):

    n_iter = 0
    for epoch in range(1, PARAMS["EPOCHS"] + 1):

        if verbose:
            pbar = tqdm(desc=f"EPOCH:{epoch}", total=len(trn_loader) + len(val_loader))

        trn_loss = 0

        for batch_idx, (X, Y) in enumerate(trn_loader):

            model.train()

            for kk in X.keys():
                X[kk] = X[kk].to(device)
            for kk in Y.keys():
                Y[kk] = Y[kk].to(device)

            # X, Y = X.to(device), Y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            Y_hat = model(X)

            Y_together = torch.cat([Y["future_1"], Y["future_2"]], dim=1)

            loss = F.mse_loss(Y_hat, Y_together)
            # weights = torch.from_numpy(np.stack([np.arange(Y.shape[1],0,-1)/np.arange(Y.shape[1],0,-1).sum()]*X.shape[0])).to(device)
            # loss = weighted_mse_loss(Y_hat, Y, weights)

            loss.backward()
            optimizer.step()

            if str(device) == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

            trn_loss += loss.detach().item()

            if verbose:
                pbar.update(1)  # PARAMS['BATCH_SIZE'])
                pbar.set_description(f"EPOCH:{epoch}, trn_loss:{trn_loss:.3f}")

            del loss, X, Y

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, (X, Y) in enumerate(val_loader):

                for kk in X.keys():
                    X[kk] = X[kk].to(device)
                for kk in Y.keys():
                    Y[kk] = Y[kk].to(device)

                Y_hat = model(X)

                Y_together = torch.cat([Y["future_1"], Y["future_2"]], dim=1)

                loss = F.mse_loss(Y_hat, Y_together)

                # weights = torch.from_numpy(np.stack([np.arange(Y.shape[1],0,-1)/np.arange(Y.shape[1],0,-1).sum()]*X.shape[0])).to(device)
                # loss = weighted_mse_loss(Y_hat, Y, weights)

                val_loss += loss.detach().item()

                if verbose:
                    pbar.update(1)  # PARAMS['BATCH_SIZE'])
                    pbar.set_description(
                        f"EPOCH:{epoch}, trn_loss:{trn_loss:.3f}, val_loss:{val_loss:.3f}"
                    )

                del loss, X, Y

        if verbose:
            pbar.close()

        if str(device) == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    # eval_visualise(model, val_loader, device, savepath)

    # whole_visualise(model, whole_loader, device, savepath)

    # eval_r2(model, test_loader, device)

    return model


def main(PARAMS):

    root = os.getcwd()

    device = torch.device(PARAMS["DEVICE"])
    print("device:", str(device))

    trn_dataset = w2w_lstm_loader(
        csv_data_path=os.path.join(
            root, "wave2web_data", "refresh_09-27", f'{PARAMS["SITE"]}_3zscore.csv'
        ),
        horizon_days=90,
        lead_days=60,
        segment=["trn"],
        site=PARAMS["SITE"],
    )
    val_dataset = w2w_lstm_loader(
        csv_data_path=os.path.join(
            root, "wave2web_data", "refresh_09-27", f'{PARAMS["SITE"]}_3zscore.csv'
        ),
        horizon_days=90,
        lead_days=60,
        segment=["val"],
        site=PARAMS["SITE"],
    )

    trn_loader = DataLoader(
        trn_dataset,
        batch_size=PARAMS["BATCH_SIZE"],
        shuffle=False,
        num_workers=PARAMS["DATALOADER_WORKERS"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=PARAMS["BATCH_SIZE"],
        shuffle=False,
        num_workers=PARAMS["DATALOADER_WORKERS"],
    )

    whole_dataset = w2w_lstm_loader(
        csv_data_path=os.path.join(
            root, "wave2web_data", "refresh_09-27", f'{PARAMS["SITE"]}_3zscore.csv'
        ),
        horizon_days=90,
        lead_days=60,
        segment=["trn", "val", "test", "deploy"],
        targets=False,
        shuffle_records=False,
        site=PARAMS["SITE"],
    )

    whole_loader = DataLoader(
        whole_dataset,
        batch_size=PARAMS["BATCH_SIZE"],
        shuffle=False,
        num_workers=PARAMS["DATALOADER_WORKERS"],
    )

    test_dataset = w2w_lstm_loader(
        csv_data_path=os.path.join(
            root, "wave2web_data", "refresh_09-27", f'{PARAMS["SITE"]}_3zscore.csv'
        ),
        horizon_days=90,
        lead_days=60,
        segment=["test"],
        targets=False,
        shuffle_records=False,
        site=PARAMS["SITE"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=PARAMS["BATCH_SIZE"],
        shuffle=False,
        num_workers=PARAMS["DATALOADER_WORKERS"],
    )

    _X, _Y = trn_dataset.__getitem__(0)

    model = Seq2Seq2Seq(
        lead_dim=PARAMS["LEAD_DIM"],
        horizon_dim=PARAMS["HORIZON_DIM"],
        hidden_dim=PARAMS["HIDDEN_DIM"],
        n_layers=PARAMS["N_LAYERS"],
        dropout=PARAMS["DROPOUT"],
        device=device,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=PARAMS["LR"])

    model = train(
        model,
        trn_loader,
        val_loader,
        optimizer,
        PARAMS,
        device=device,
        savepath=None,
        verbose=True,
    )

    torch.save(
        model.state_dict(),
        os.path.join(
            os.getcwd(),
            "w2w_results",
            "refresh_09-27",
            f"{whole_loader.dataset.site}_weights.pth",
        ),
    )

    test_output(model, test_loader, device)

    whole_visualise_and_output(model, whole_loader, device)

    perturb(0.50, model, device)

    perturb(-0.50, model, device)


def perturb(pert_amt, model, device):

    print("perturbing", pert_amt)

    pert_dataset = w2w_lstm_loader(
        csv_data_path=os.path.join(
            os.getcwd(),
            "wave2web_data",
            "refresh_09-27",
            f'{PARAMS["SITE"]}_3zscore.csv',
        ),
        horizon_days=90,
        lead_days=60,
        segment=["trn", "val", "test", "deploy"],
        perturb=pert_amt,
        targets=False,
        shuffle_records=False,
        site=PARAMS["SITE"],
    )

    pert_loader = DataLoader(
        pert_dataset,
        batch_size=PARAMS["BATCH_SIZE"],
        shuffle=False,
        num_workers=PARAMS["DATALOADER_WORKERS"],
    )

    # collate results
    model.eval()
    Y_hat_results = []
    with torch.no_grad():
        val_loss = 0
        for batch_idx, X in enumerate(pert_loader):

            for kk in X.keys():
                X[kk] = X[kk].to(device)

            Y_hat = model(X)

            Y_hat_results.append(Y_hat.detach().to("cpu").numpy())

    data = np.concatenate(Y_hat_results)

    results_df = pd.DataFrame(
        index=list(pert_loader.dataset.records.values()),  # datetimes
        data=data,
        columns=[
            timedelta(days=ii)
            for ii in list(range(1, pert_loader.dataset.horizon_days + 1))
        ],
    )

    results_df = results_df.sort_index()

    results_df.to_csv(
        os.path.join(
            os.getcwd(),
            "w2w_results",
            "refresh_09-27",
            pert_loader.dataset.site + f"p{pert_amt}_full_result.csv",
        )
    )


def test_output(model, test_loader, device):

    # collate results
    model.eval()
    Y_hat_results = []
    with torch.no_grad():
        val_loss = 0
        for batch_idx, X in enumerate(test_loader):

            for kk in X.keys():
                X[kk] = X[kk].to(device)

            Y_hat = model(X)

            Y_hat_results.append(Y_hat.detach().to("cpu").numpy())

    data = np.concatenate(Y_hat_results)

    results_df = pd.DataFrame(
        index=list(test_loader.dataset.records.values()),  # datetimes
        data=data,
        columns=[
            timedelta(days=ii)
            for ii in list(range(1, test_loader.dataset.horizon_days + 1))
        ],
    )

    results_df = results_df.sort_index()

    results_df = results_df.unstack().reset_index()

    results_df["prediction_dt"] = results_df["level_0"] + results_df["level_1"]

    results_df = results_df.loc[
        results_df["level_0"].isin(
            [timedelta(days=ii) for ii in [1, 5, 15, 30, 60, 90]]
        )
    ]

    results_df = (
        results_df.set_index(["level_0", "prediction_dt"])
        .drop(columns="level_1")
        .unstack("level_0")
    )

    results_df.columns = results_df.columns.droplevel()
    results_df.columns = [str(el).replace(" 00:00:00", "") for el in results_df.columns]

    results_df = pd.merge(
        results_df,
        test_loader.dataset.df[["volume_bcm"]],
        how="left",
        left_index=True,
        right_index=True,
    )

    rscores = {
        f"{ii}_days": r2_score(
            results_df.loc[
                ~results_df[["volume_bcm", f"{ii} days"]].isna().any(axis=1),
                "volume_bcm",
            ],
            results_df.loc[
                ~results_df[["volume_bcm", f"{ii} days"]].isna().any(axis=1),
                f"{ii} days",
            ],
        )
        for ii in [1, 5, 15, 30, 60, 90]
    }

    print(test_loader.dataset.site, rscores)

    json.dump(
        rscores,
        open(
            os.path.join(
                os.getcwd(),
                "w2w_results",
                "refresh_09-27",
                test_loader.dataset.site + "_r2score.json",
            ),
            "w",
        ),
    )


def whole_visualise_and_output(model, whole_loader, device):

    # collate results
    model.eval()
    Y_hat_results = []
    with torch.no_grad():
        val_loss = 0
        for batch_idx, X in enumerate(whole_loader):

            for kk in X.keys():
                X[kk] = X[kk].to(device)

            Y_hat = model(X)

            Y_hat_results.append(Y_hat.detach().to("cpu").numpy())

    data = np.concatenate(Y_hat_results)

    results_df = pd.DataFrame(
        index=list(whole_loader.dataset.records.values()),  # datetimes
        data=data,
        columns=[
            timedelta(days=ii)
            for ii in list(range(1, whole_loader.dataset.horizon_days + 1))
        ],
    )

    results_df = results_df.sort_index()

    results_df.to_csv(
        os.path.join(
            os.getcwd(),
            "w2w_results",
            "refresh_09-27",
            whole_loader.dataset.site + "_full_result.csv",
        )
    )

    results_df = results_df.unstack().reset_index()

    results_df["prediction_dt"] = (
        results_df["level_0"] + results_df["level_1"]
    )  # date + timedelta

    summary_df = results_df.loc[
        results_df["level_0"].isin([timedelta(days=ii) for ii in [30, 60, 90]])
    ]

    summary_df = (
        summary_df.set_index(["level_0", "prediction_dt"])
        .drop(columns="level_1")
        .unstack("level_0")
    )
    summary_df.columns = summary_df.columns.droplevel()
    summary_df.columns = [str(el).replace(" 00:00:00", "") for el in summary_df.columns]

    summary_df = pd.merge(
        summary_df,
        whole_loader.dataset.df[["volume_bcm"]],
        how="left",
        left_index=True,
        right_index=True,
    )

    plot = summary_df.rename(columns={"volume_bcm": "actual"}).plot(figsize=(20, 6))
    fig = plot.get_figure()
    fig.savefig(os.path.join(os.getcwd(), whole_loader.dataset.site + "_30_60_90.png"))


if __name__ == "__main__":

    for kk in [
        "kabini",
        "harangi",
        "hemavathy",
        "krishnaraja_sagar",
        "bhadra",
        "lower_bhawani",
    ]:

        PARAMS = dict(
            SITE=kk,
            BATCH_SIZE=256,
            DATALOADER_WORKERS=8,
            LR=0.05,
            DEVICE="cuda",
            EPOCHS=40,
            N_LAYERS=2,
            HIDDEN_DIM=8,
            LEAD_DIM=60,
            HORIZON_DIM=90,
            DROPOUT=0.5,
        )

        main(PARAMS)
