from h2ox.ai.dataset.dataset_factory import DatasetFactory
from ruamel.yaml import YAML
from pathlib import Path


if __name__ == "__main__":
    yml_path = Path("conf.yaml")
    with yml_path.open('r') as fp:
        yaml = YAML(typ="safe")
        cfg = yaml.load(fp)

    # update the cfg paths in the testing code
    cfg["data_parameters"]["cache_path"] = Path(".").absolute() / "data/cache.nc"
    cfg["data_parameters"]["data_units"]["forecast"]["gdf_path"] = Path(".").absolute() / "data/basins.geojson"
    cfg["data_parameters"]["data_units"]["historic"]["gdf_path"] = Path(".").absolute() / "data/basins.geojson"

    dsf = DatasetFactory(cfg)
    dsf.build_dataset()
    assert False

    