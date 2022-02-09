from pathlib import Path
from ruamel.yaml import YAML
import xarray as xr
from torch.utils.data import DataLoader
from h2ox.ai.dataset.dataset import FcastDataset
from h2ox.ai.train import train_validation_test_split
from h2ox.ai.dataset.utils import load_zscore_data


def create_dummy_data() -> xr.Dataset:
    """Create dummy dataset with same shape as output of dataset factory
    Dimensions: 
        steps: timedelta64[ns] = 90
        date: datetime64[ns] = 1000
        global_sites: str = 6
    
    Variables:
        historic_t2m: float64 = (steps, date, global_sites)
        historic_tp: float64 = (steps, date, global_sites)
        forecast_t2m: float64 = (steps, date, global_sites)
        forecast_tp: float64 = (steps, date, global_sites)
        targets_WATER_VOLUME: float64 = (steps, date, global_sites)
        doy_sin: float64 = (steps, date, global_sites)
    """

    pass


if __name__ == "__main__":
    yml_path = Path("tests/test_conf.yaml")
    with yml_path.open('r') as fp:
        yaml = YAML(typ="safe")
        cfg = yaml.load(fp)

    data_dir = Path(Path.cwd() / "data")
    if (data_dir / "cache.nc").exists():
        ds = xr.open_dataset(data_dir / "cache.nc")
    else:
        ds = create_dummy_data()
    
    data = FcastDataset(
        ds,
        **cfg["dataset_parameters"],
    )

    train_dd, val_dd, test_dd = train_validation_test_split(
        data, 
        cfg=cfg,
        time_dim="date"
    )

    meta_df = train_dd.dataset._get_meta_dataframe()
    train_samples = meta_df.loc[train_dd.indices]
    val_samples = meta_df.loc[val_dd.indices]
    test_samples = meta_df.loc[test_dd.indices]

    train_eg = train_dd[0]
    val_eg = val_dd[0]
    test_eg = test_dd[0]

    test_dl = DataLoader(test_dd, batch_size=111)
    dummy = test_dl.__iter__().__next__()
    print(dummy["meta"]["site"].shape)

    assert False
