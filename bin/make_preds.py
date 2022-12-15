import json
import os

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from h2ox.ai.dataset import DatasetFactory, maybe_load
from h2ox.ai.dataset.dataset import train_validation_test_split
from h2ox.ai.dataset.utils import calculate_errors
from h2ox.ai.model_gnn import initialise_gnn
from h2ox.ai.train_bayesian import test


def datasets_only(
    dataset_parameters: dict,
    data_parameters: dict,
    model_parameters: dict,
    training_parameters: dict,
):

    dd = DatasetFactory(
        {"data_parameters": data_parameters, "dataset_parameters": dataset_parameters}
    ).build_dataset()

    if dataset_parameters["norm_difference"]:
        dd.augment_dict
        dd.target_var[0]

        # _std_target = dict(
        #     zip(
        #        var_norms["std_norm"]["shift_targets_WATER_VOLUME"]["std"].to_dict()[
        #            "coords"
        #        ]["global_sites"]["data"],
        #        var_norms["std_norm"]["shift_targets_WATER_VOLUME"]["std"].to_dict()[
        #            "data"
        #        ],
        #    )
        # )
    else:
        pass

    # train-validation split
    train_dd, validation_dd, test_dd = train_validation_test_split(
        dd,
        cfg=dataset_parameters,
        time_dim="date",
    )

    return len(train_dd.indices), len(validation_dd.indices), len(test_dd.indices)


def test_loop(
    dataset_parameters: dict,
    data_parameters: dict,
    model_parameters: dict,
    training_parameters: dict,
    state_dict_path: str,
) -> int:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if "graph_conv" not in model_parameters.keys():
        model_parameters["graph_conv"] = True

    dd = DatasetFactory(
        {"data_parameters": data_parameters, "dataset_parameters": dataset_parameters}
    ).build_dataset()

    if dataset_parameters["norm_difference"]:
        var_norms = dd.augment_dict
        target_var = dd.target_var[0]

        std_target = dict(
            zip(
                var_norms["std_norm"]["shift_targets_WATER_VOLUME"]["std"].to_dict()[
                    "coords"
                ]["global_sites"]["data"],
                var_norms["std_norm"]["shift_targets_WATER_VOLUME"]["std"].to_dict()[
                    "data"
                ],
            )
        )
    else:
        var_norms = None
        target_var = None

    # train-validation split
    train_dd, validation_dd, test_dd = train_validation_test_split(
        dd,
        cfg=dataset_parameters,
        time_dim="date",
    )

    test_dl = DataLoader(
        test_dd,
        batch_size=training_parameters["batch_size"],
        shuffle=False,
        num_workers=training_parameters["num_workers"],
    )

    item = dd.__getitem__(0)

    model = initialise_gnn(
        item,
        sites=maybe_load(dataset_parameters["select_sites"]),
        sites_edges=maybe_load(dataset_parameters["sites_edges"]),
        flow_std=std_target,
        device=device,
        graph_conv=model_parameters["graph_conv"],
        hidden_size=model_parameters["hidden_size"],
        num_layers=model_parameters["num_layers"],
        dropout=model_parameters["dropout"],
        bayesian_linear=model_parameters["bayesian_linear"],
        bayesian_lstm=model_parameters["bayesian_lstm"],
        lstm_params=model_parameters["lstm_params"],
    )

    # force float
    model = model.to(torch.float)

    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    preds = test(model, test_dl, denorm=var_norms, denorm_var=target_var)

    errors = calculate_errors(
        preds,
        obs_var="obs",
        sim_var="sim" if model_parameters["model_str"] == "s2s2s" else "sim-frozen",
        site_dim="site",
    )

    return preds, errors


def check():
    sites_paths = yaml.load(
        open(os.path.join(os.getcwd(), "bin", "./experiments.yaml")),
        Loader=yaml.SafeLoader,
    )

    os.path.join(os.getcwd(), "data", "final_preds")

    for site in sites_paths.keys():
        for exp in ["gconv", "no_gconv"]:

            logger.info(f"Running {site} - {exp}")

            state_dict_path = os.path.join(
                os.getcwd(), "/".join(sites_paths[site][exp].split("/")[1:])
            )
            dir_root = os.path.split(state_dict_path)[0]
            cfg_path = os.path.join(dir_root, "config.json")

            print(os.path.exists(cfg_path))
            print(os.path.exists(state_dict_path))


def datasets_check():

    sites_paths = yaml.load(
        open(os.path.join(os.getcwd(), "bin", "./experiments.yaml")),
        Loader=yaml.SafeLoader,
    )

    outpath = os.path.join(os.getcwd(), "data", "final_preds")

    ds_lens = {}

    for site in sites_paths.keys():

        exp = "gconv"

        if site not in ["extras", "penner"]:

            state_dict_path = os.path.join(
                os.getcwd(), "/".join(sites_paths[site][exp].split("/")[1:])
            )
            dir_root = os.path.split(state_dict_path)[0]
            cfg = json.load(open(os.path.join(dir_root, "config.json")))

            trn, val, test = datasets_only(
                cfg["dataset_parameters"],
                cfg["data_parameters"],
                cfg["model_parameters"],
                cfg["training_parameters"],
            )

            ds_lens[site] = {
                "trn": trn,
                "val": val,
                "test": test,
            }

    print(ds_lens)

    json.dump(ds_lens, open(os.path.join(outpath, "ds_lens.json"), "w"))


def main(val_lowest=False):

    sites_paths = yaml.load(
        open(os.path.join(os.getcwd(), "bin", "./experiments.yaml")),
        Loader=yaml.SafeLoader,
    )

    outpath = os.path.join(os.getcwd(), "data", "final_preds")

    for site in sites_paths.keys():

        if site in ["extras", "penner"]:
            for exp in ["gconv"]:

                logger.info(f"Running {site} - {exp}")

                state_dict_path = os.path.join(
                    os.getcwd(), "/".join(sites_paths[site][exp].split("/")[1:])
                )
                dir_root = os.path.split(state_dict_path)[0]
                cfg = json.load(open(os.path.join(dir_root, "config.json")))

                if val_lowest:
                    metrics = json.load(open(os.path.join(dir_root, "metrics.json")))

                    idx = np.array(metrics["Loss/val"]["values"]).argmin()

                    step = metrics["Loss/val"]["steps"][idx]

                    step = str(int(step) + 10)

                    state_dict_path = os.path.join(dir_root, f"model_epoch0{step}.pt")

                    assert os.path.exists(
                        state_dict_path
                    ), f"path DNE {state_dict_path}"

                    logger.info(f"VAL LOWST: for {idx}, {step}")

                if val_lowest:
                    if os.path.exists(
                        os.path.join(outpath, f"{site}-{exp}-{step}-preds.nc")
                    ):
                        logger.info(
                            f'FOUND {os.path.join(outpath,f"{site}-{exp}-{step}-preds.nc")}'
                        )
                        continue

                preds, errors = test_loop(
                    cfg["dataset_parameters"],
                    cfg["data_parameters"],
                    cfg["model_parameters"],
                    cfg["training_parameters"],
                    state_dict_path,
                )

                if val_lowest:
                    errors.to_netcdf(
                        os.path.join(outpath, f"{site}-{exp}-{step}-errors.nc")
                    )
                    preds.to_netcdf(
                        os.path.join(outpath, f"{site}-{exp}-{step}-preds.nc")
                    )
                else:

                    errors.to_netcdf(os.path.join(outpath, f"{site}-{exp}-errors.nc"))
                    preds.to_netcdf(os.path.join(outpath, f"{site}-{exp}-preds.nc"))


if __name__ == "__main__":
    main(val_lowest=True)
