import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from ts.torch_handler.base_handler import BaseHandler

from h2ox.ai.model_gnn import initialise_gnn

# pipe logger to sys.stdout specifically - no error in cloud console
logger.remove()
logger.add(sys.stdout, colorize=False, format="{time:YYYYMMDDHHmmss}|{level}|{message}")


class H2OxHandler(BaseHandler):
    """
    The handler takes an input string and returns the classification text
    based on the serialized transformers checkpoint.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Loads the model.pt file and initializes the model object.
        Loads preprocessing vars
        Loads labels to name mapping file for post-processing inference response
        """
        self.manifest = ctx.manifest

        # todo: get these from env.
        self.approach = os.environ.get("SAMPLE_METHOD", "sample-paths")
        self.n_samples = os.environ.get("MC_N_SAMPLES", 10)
        self.std_scale = os.environ.get("STD_SCALE", 2)

        properties = ctx.system_properties
        print(properties)
        model_dir = properties.get("model_dir")
        model_name = os.path.splitext(
            os.path.split(self.manifest["model"]["serializedFile"])[-1]
        )[0]
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or pytorch_model.bin file")

        sites = json.load(open(os.path.join(model_dir, f"{model_name}_sites.json")))
        edges = json.load(open(os.path.join(model_dir, "all_edges.json")))
        run_cfg = json.load(open(os.path.join(model_dir, f"{model_name}_cfg.json")))
        self.var_norm = json.load(
            open(os.path.join(model_dir, f"{model_name}_preprocessing.json"))
        )
        self.sites = sites
        model_parameters = run_cfg["model_parameters"]
        item = pickle.load(
            open(os.path.join(model_dir, f"{model_name}_dummy_item.pkl"), "rb")
        )

        flow_std = self.var_norm["std_norm"]["shift_targets_WATER_VOLUME"]["std"]

        # initialise model
        """ !!! Why doesn't this work??? !!!
        self.model = LayeredBayesianGraphLSTM(
            sites=sites,
            sites_edges=edges,
            flow_std=flow_std,
            future_horizon=run_cfg["dataset_parameters"]["future_horizon"],
            forecast_horizon=run_cfg["dataset_parameters"]["forecast_horizon"],
            historic_period=run_cfg["dataset_parameters"]["historical_seq_len"],
            historical_input_size=len(run_cfg["dataset_parameters"]["historic_variables"])-1,
            forecast_input_size=len(run_cfg["dataset_parameters"]["forecast_variables"]),
            future_input_size=len(run_cfg["dataset_parameters"]["future_variables"]),
            hidden_size=run_cfg["model_parameters"]["hidden_size"],
            target_size=run_cfg["dataset_parameters"]["forecast_horizon"]+run_cfg["dataset_parameters"]["future_horizon"],
            digraph = run_cfg["model_parameters"]["digraph"],
            diag = run_cfg["model_parameters"]["daig"],
            num_layers = run_cfg["model_parameters"]["num_layers"],
            dropout = run_cfg["model_parameters"]["dropout"],
            bayesian_lstm = run_cfg["model_parameters"]["bayesian_lstm"],
            bayesian_linear = run_cfg["model_parameters"]["bayesian_linear"],
            device = self.device,
            lstm_params = run_cfg["model_parameters"]["lstm_params"],
            # include_current_timestep_in_horizon: bool = True,
        ).float()
        """

        self.model = initialise_gnn(
            item,
            sites=sites,
            sites_edges=edges,
            flow_std=flow_std,
            device=self.device,
            hidden_size=model_parameters["hidden_size"],
            num_layers=model_parameters["num_layers"],
            dropout=model_parameters["dropout"],
            bayesian_linear=model_parameters["bayesian_linear"],
            bayesian_lstm=model_parameters["bayesian_lstm"],
            lstm_params=model_parameters["lstm_params"],
        )

        self.cfg = run_cfg

        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))

        # self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        # self.model.eval()
        logger.info(f"Model from path {model_dir} loaded successfully")

        self.initialized = True

    def preprocess(self, data):
        """Preprocessing input request by tokenizing
        Extend with your own preprocessing steps as needed
        """
        # shift
        # data = copy.deepcopy(data)

        if self.cfg["dataset_parameters"]["variables_difference"] is not None:
            for var in self.cfg["dataset_parameters"]["variables_difference"]:
                for date in data.keys():
                    for site in data[date].keys():
                        for horizon in ["x_d", "x_f", "x_ff", "y"]:
                            if var in data[date][site][horizon].keys():
                                data[date][site][horizon][var] = [0] + (
                                    np.array(data[date][site][horizon][var][1:])
                                    - np.array(data[date][site][horizon][var][0:-1])
                                ).tolist()

        # norm
        if self.cfg["dataset_parameters"]["normalise"] is not None:
            for var in self.cfg["dataset_parameters"]["normalise"]:
                for date in data.keys():
                    for site in data[date].keys():
                        for horizon in ["x_d", "x_f", "x_ff", "y"]:
                            if var in data[date][site][horizon].keys():
                                data[date][site][horizon][var] = (
                                    (
                                        np.array(data[date][site][horizon][var])
                                        - self.var_norm["normalise"][var]["min"][site]
                                    )
                                    / (
                                        self.var_norm["normalise"][var]["max"][site]
                                        - self.var_norm["normalise"][var]["min"][site]
                                    )
                                ).tolist()

        # stdnorm
        if self.cfg["dataset_parameters"]["std_norm"] is not None:
            for var in self.cfg["dataset_parameters"]["std_norm"]:
                for date in data.keys():
                    for site in data[date].keys():
                        for horizon in ["x_d", "x_f", "x_ff", "y"]:
                            if var in data[date][site][horizon].keys():
                                data[date][site][horizon][var] = (
                                    np.array(data[date][site][horizon][var])
                                    / self.var_norm["std_norm"][var]["std"][site]
                                ).tolist()

        # zscore
        if self.cfg["dataset_parameters"]["zscore"] is not None:
            for var in self.cfg["dataset_parameters"]["zscore"]:
                for date in data.keys():
                    for site in data[date].keys():
                        for horizon in ["x_d", "x_f", "x_ff", "y"]:
                            if var in data[date][site][horizon].keys():
                                data[date][site][horizon][var] = (
                                    (
                                        np.array(data[date][site][horizon][var])
                                        - self.var_norm["zscore"][var]["mean"][site]
                                    )
                                    / self.var_norm["zscore"][var]["std"][site]
                                ).tolist()

        # build inputs
        inputs = {}

        for date, vals in data.items():
            sample = {}

            for site in vals.keys():

                sample[site] = {
                    "x_d": torch.from_numpy(
                        np.array(
                            [
                                [
                                    vals[site]["x_d"][var]
                                    for var in self.cfg["dataset_parameters"][
                                        "historic_variables"
                                    ]
                                    if var
                                    not in self.cfg["dataset_parameters"]["target_var"]
                                ]
                            ],
                            dtype=np.float32,
                        )
                    )
                    .to(self.device)
                    .permute(0, 2, 1)[:, 1:, :],
                    "x_f": torch.from_numpy(
                        np.array(
                            [
                                [
                                    vals[site]["x_f"][var]
                                    for var in self.cfg["dataset_parameters"][
                                        "forecast_variables"
                                    ]
                                ]
                            ],
                            dtype=np.float32,
                        )
                    )
                    .to(self.device)
                    .permute(0, 2, 1)[:, 1:, :],
                    "x_ff": torch.from_numpy(
                        np.array(
                            [
                                [
                                    vals[site]["x_ff"][var]
                                    for var in self.cfg["dataset_parameters"][
                                        "future_variables"
                                    ]
                                ]
                            ],
                            dtype=np.float32,
                        )
                    )
                    .to(self.device)
                    .permute(0, 2, 1),
                    "hist_level": torch.from_numpy(
                        np.array(
                            [
                                [
                                    vals[site]["y"][var]
                                    for var in self.cfg["dataset_parameters"][
                                        "target_var"
                                    ]
                                ]
                            ],
                            dtype=np.float32,
                        )
                    )
                    .to(self.device)
                    .permute(0, 2, 1)[:, 1:, :],
                }

                # forward fill nans in hist_level only to allow inference
                sample[site]["hist_level"] = torch.from_numpy(
                    np.expand_dims(
                        pd.DataFrame(sample[site]["hist_level"].numpy().squeeze(0))
                        .fillna(method="ffill", axis=0)
                        .values,
                        0,
                    )
                )

            inputs[date] = sample

        return inputs

    def inference(self, inputs):
        """Predict the class of a text using a trained transformer model."""

        predictions = {}
        for date, sample in inputs.items():

            sample_predictions = []
            for _ii in range(self.n_samples):
                print(_ii)

                prediction = self.model(sample)

                sample_predictions.append(prediction.squeeze())

            predictions[date] = torch.stack(sample_predictions)

        logger.info("Model predicted")
        return predictions

    def postprocess(self, inference_output, inputs):
        def reverse_cumsum(arr, dim):
            shp = list(arr.shape)
            shp[dim] = 1
            return torch.diff(
                arr, dim=dim, prepend=torch.zeros(*list(shp)).to(arr.device)
            )

        # optionally reshape
        for _date in inference_output.keys():
            if len(inference_output[_date].shape) == 2:
                inference_output[_date] = inference_output[_date].unsqueeze(-1)

        # target_difference
        if self.cfg["dataset_parameters"]["target_difference"]:

            if self.cfg["dataset_parameters"]["norm_difference"]:
                # first denorm
                for _date, data in inference_output.items():
                    for ii_s, site in enumerate(self.sites):

                        data[:, :, ii_s] = (
                            data[:, :, ii_s]
                            * self.var_norm["std_norm"]["shift_targets_WATER_VOLUME"][
                                "std"
                            ][site]
                        )

            results = {}

            # then convert back to levels
            hist_levels = {}
            for date in inference_output.keys():
                hist_levels[date] = torch.stack(
                    [
                        inputs[date][site]["hist_level"][:, -1, :].squeeze()
                        for site in self.sites
                    ]
                )

            if self.approach == "paths-sort":
                for date, data in inference_output.items():
                    paths = torch.cumsum(data, dim=1)

                    results[date] = {
                        "upper": paths.sort(dim=0).values[-1, :, :] + hist_levels[date],
                        "lower": paths.sort(dim=0).values[0, :, :] + hist_levels[date],
                        "mean": paths.mean(dim=0) + hist_levels[date],
                    }  # levels 0-1

            elif self.approach == "sample-diffs":

                for date, data in inference_output.items():
                    mu = data.mean(dim=0)
                    sigma = data.std(dim=0)

                    results[date] = {
                        "upper": (mu + self.std_scale * sigma).cumsum(dim=0)
                        + hist_levels[date],
                        "lower": (mu - self.std_scale * sigma).cumsum(dim=0)
                        + hist_levels[date],
                        "mean": mu.cumsum(dim=0) + hist_levels[date],
                    }

            elif self.approach == "sample-paths":

                for date, data in inference_output.items():

                    paths = torch.cumsum(data, dim=1)
                    mu = paths.mean(dim=0)
                    sigma = paths.std(dim=0)

                    results[date] = {
                        "upper": (mu + self.std_scale * sigma) + hist_levels[date],
                        "lower": (mu - self.std_scale * sigma) + hist_levels[date],
                        "mean": mu + hist_levels[date],
                    }

            elif self.approach == "diffs-sort":

                for date, data in inference_output.items():

                    sort_stack = data.sort(dim=0).values.cumsum(dim=1)

                    paths = torch.cumsum(data, dim=1)
                    mu = paths.mean(dim=0)
                    sigma = paths.std(dim=0)

                    results[date] = {
                        "upper": sort_stack[-1, :, :] + hist_levels[date],
                        "lower": sort_stack[0, :, :] + hist_levels[date],
                        "mean": sort_stack.mean(dim=0) + hist_levels[date],
                    }

            else:
                raise NotImplementedError

            # clip back to the norm range of 0-1
            for date in results.keys():
                results[date]["upper"] = results[date]["upper"].clip(0.0, 1.0)
                results[date]["lower"] = results[date]["lower"].clip(0.0, 1.0)
                results[date]["mean"] = results[date]["mean"].clip(0.0, 1.0)

            # finally de-norm the results
            stretch = torch.from_numpy(
                np.array(
                    [
                        (
                            self.var_norm["normalise"]["targets_WATER_VOLUME"]["max"][
                                site
                            ]
                            - self.var_norm["normalise"]["targets_WATER_VOLUME"]["min"][
                                site
                            ]
                        )
                        for site in self.sites
                    ]
                )
            ).to(self.device)
            mins = torch.from_numpy(
                np.array(
                    [
                        self.var_norm["normalise"]["targets_WATER_VOLUME"]["min"][site]
                        for site in self.sites
                    ]
                )
            ).to(self.device)

            for date in results.keys():
                results[date]["upper"] = results[date]["upper"] * stretch + mins
                results[date]["lower"] = results[date]["lower"] * stretch + mins
                results[date]["mean"] = results[date]["mean"] * stretch + mins

            for date in results.keys():
                for band in results[date].keys():
                    results[date][band] = results[date][band].detach().numpy().tolist()

        else:
            raise NotImplementedError

        return [results]

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output, model_input)


if __name__ == "__main__":

    from ts.context import Context

    ctx = Context(
        model_name="extras",
        model_dir="/home/jupyter/PIPELINE/ts/h2ox-ai/models",
        manifest={
            "model": {
                "serializedFile": "extras.pt",
            }
        },
        batch_size=8,
        gpu=None,
        mms_version=0.1,
        limit_max_image_pixels=True,
    )

    inst = H2OxHandler()

    inst.initialize(ctx)

    sample_data = [pickle.load(open("./extras_sample.pkl", "rb"))]
    # sample_data = json.load(open("./data/kaveri_sample.json"))

    inps = inst.preprocess(sample_data)

    print(inps)
    y_hat = inst.inference(inps)

    print("yhat", y_hat)
    print("SHAPE")
    print([(kk, vv.shape) for kk, vv in y_hat.items()])

    results = inst.postprocess(y_hat, inps)

    print("RESULTS")
    print(results)
