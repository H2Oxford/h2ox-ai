import os
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from random import shuffle

from torch.utils.data import DataLoader, Dataset


class w2w_dnn_loader(Dataset):
    """
    Some Docstrings

    """

    def __init__(
        self,
        csv_data_path,
        targets_forecast,
        lag_windows,
        segment,
        normalise,
        autoregressive,
        start_date=None,
        end_date=None,
        target_var: str = "PRESENT_STORAGE_TMC",
    ):

        self.segment = segment
        self.lag_windows = lag_windows
        self.targets_forecast = targets_forecast
        self.autoregressive = autoregressive

        ### load the data
        df = pd.read_csv(csv_data_path).set_index("Unnamed: 0")
        df.index = pd.to_datetime(df.index)

        if normalise:
            self.Y_mean = df[target_var].mean()
            self.Y_std = df[target_var].std()
            df[target_var] = (df[target_var] - df[target_var].mean()) / df[
                target_var
            ].std()

        ### construct the features
        df["Y"] = df[target_var]
        df["Y_1"] = df[target_var].shift(-1)

        for ii in range(5, targets_forecast, 5):
            df[f"Y_{ii}"] = df[target_var].shift(-1 * ii)

        # maybe normalise
        if normalise:
            for variable in ["tp", "t2m"]:
                df[variable] = (df[variable] - df[variable].mean()) / df[variable].std()

                for step in range(15):
                    df[f"{variable}_{step}"] = (
                        df[f"{variable}_{step}"] - df[f"{variable}_{step}"].mean()
                    ) / df[f"{variable}_{step}"].std()

        ### add lagged feature
        for window in lag_windows:
            for variable in ["tp", "t2m"]:
                df[f"lag_{variable}_{window}"] = df[variable].rolling(window).mean()

        ### add day-of-year cosine
        df["sin_dayofyear"] = np.sin((df.index.dayofyear - 1) / 365 * 2 * np.pi)

        ### attach to instance
        self.df = df

        ### set up the records -> a dict with int_index:shuffled(dt_index)

        valid_idxs = (
            (~pd.isna(df[target_var]))
            & (~pd.isna(df["tp"]))
            & (~pd.isna(df["t2m"]))
            & (~(pd.isna(df[[f"tp_{ii}" for ii in range(15)]]).any(axis=1)))
            & (~pd.isna(df[[f"t2m_{ii}" for ii in range(15)]]).any(axis=1))
            & (
                ~pd.isna(
                    df[[f"Y_{ii}" for ii in range(5, self.targets_forecast, 5)]]
                ).any(axis=1)
            )
        )

        if segment is not None:
            valid_idxs = valid_idxs & (df["segment"] == segment)

        if start_date is not None:
            valid_idxs = valid_idxs & (df.index >= start_date) & (df.index < end_date)

        print("valid records:", valid_idxs.sum())
        self.records = df.loc[valid_idxs].index.tolist()

        shuffle(self.records)
        self.records = dict(zip(range(len(self.records)), self.records))

    def __len__(self):
        return len(self.records.keys())

    def __getitem__(self, index):

        idx_dt = self.records[index]

        # Y -> 1, 5, etc. timesteps in the future
        Y = self.df.loc[
            idx_dt, ["Y_1"] + [f"Y_{ii}" for ii in range(5, self.targets_forecast, 5)]
        ].values

        # X -> Y, X_, lagged(X_), sin_dayofyear, forecast
        X_columns = (
            ["tp", "t2m"]
            + [
                f"lag_{variable}_{window}"
                for window in self.lag_windows
                for variable in ["tp", "t2m"]
            ]
            + ["sin_dayofyear"]
            + [f"tp_{ii}" for ii in range(1, 14)]
            + [f"t2m_{ii}" for ii in range(1, 14)]
        )

        if self.autoregressive:
            X_columns = ["Y"] + X_columns

        X = self.df.loc[idx_dt, X_columns].values
        # X = np.random.rand(1).reshape((1,))

        # if np.isnan(X.astype(np.float32)).sum()>0:
        #    print ('X_NANNN', idx_dt)
        #    print (X)
        # if np.isnan(Y.astype(np.float32)).sum()>0:
        #    print ('Y_NANNN', idx_dt)
        #    print (Y)

        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(
            Y.astype(np.float32)
        )


if __name__ == "__main__":
    ### do some tests

    root = os.getcwd()

    loader = w2w_dnn_loader(
        csv_data_path=os.path.join(root, "wave2web_data", "Kabini_12x3mo_split.csv"),
        targets_forecast=90,  # days
        lag_windows=[5],  # days
        segment=None,
        normalise=True,
        autoregressive=False,
    )

    for ii in np.random.choice(loader.__len__(), 10):
        X, Y = loader.__getitem__(ii)
        print(ii, X, Y)
        print(X.shape, Y.shape)

    for ii in range(loader.__len__()):
        X, Y = loader.__getitem__(ii)
