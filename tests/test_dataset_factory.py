from typing import Dict, Any
from h2ox.ai.dataset.dataset_factory import DatasetFactory
from ruamel.yaml import YAML
from pathlib import Path
from copy import deepcopy


def get_single_dataunit_dict(cfg, key_to_keep: str):
    cfg_cp = deepcopy(cfg)

    keys_to_remove = [k for k in cfg_cp["data_parameters"]["data_units"].keys() if k != key_to_keep]
    for key in keys_to_remove:
        del cfg_cp["data_parameters"]["data_units"][key]

    return cfg_cp


def test_individual_build_data_units(cfg: Dict[str, Any]):
    # subset of data
    doy_cfg = get_single_dataunit_dict(cfg, "doy") 
    forecast_cfg = get_single_dataunit_dict(cfg, "forecast")
    history_cfg = get_single_dataunit_dict(cfg, "history")
    targets_cfg = get_single_dataunit_dict(cfg, "targets")
    
    # check the dataset construction for each data unit
    for cfg_ in [targets_cfg, doy_cfg, forecast_cfg, history_cfg]:
        cfg_type = [str(k) for k in cfg_["data_parameters"]["data_units"].keys()][0]
        dsf = DatasetFactory(cfg_)
        arrays = dsf._build_data()
        
        if cfg_type == "forecast":
            assert False


def test_creation_of_all_test_data(cfg: Dict[str, Any]):
    dsf = DatasetFactory(cfg)
    fcast_ds = dsf.build_dataset()
    assert False


if __name__ == "__main__":
    yml_path = Path("tests/test_conf.yaml")
    with yml_path.open('r') as fp:
        yaml = YAML(typ="safe")
        cfg = yaml.load(fp)

    # update the cfg paths in the testing code
    cfg["data_parameters"]["cache_path"] = Path(".").absolute() / "data/cache.nc"
    cfg["data_parameters"]["data_units"]["forecast"]["gdf_path"] = Path(".").absolute() / "data/basins.geojson"
    cfg["data_parameters"]["data_units"]["historic"]["gdf_path"] = Path(".").absolute() / "data/basins.geojson"
        
    test_creation_of_all_test_data(cfg)