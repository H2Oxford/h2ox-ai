from typing import Dict, Any, Union
import os
from datetime import datetime
from pydoc import locate
import collections
from pathlib import Path
import xarray as xr
import yaml
from loguru import logger
from torch.utils.data import Dataset


# def convert_pathlib_opts_to_str(data: Dict[str, Union[Any, Dict]]):
#     # https://stackoverflow.com/a/1254499/9940782
#     if isinstance(data, Path):
#         return data.as_posix()
#     elif isinstance(data, collections.Mapping):
#         return dict(map(convert, data.iteritems()))
#     elif isinstance(data, collections.Iterable):
#         return type(data)(map(convert, data))
#     else:
#         return data


class DatasetFactory:
    def __init__(
        self,
        cfg: Dict[str, Any],
    ):
        self.cfg = cfg["data_parameters"]
        self.ptds_cfg = cfg["dataset_parameters"]

    @staticmethod
    def get_site_mapper(data_unit_site_keys, global_site_keys):
        """a placeholder method just to map site keys for each DataUnit to global site keys"""
        return dict(zip(data_unit_site_keys, global_site_keys))

    def check_cache(self):

        # if exists
        if os.path.exists(self.cfg["cache_path"]):
            # load cfg
            root, ext = os.path.splitext(self.cfg["cache_path"])

            cache_cfg = yaml.load(open(root + ".yaml"), Loader=yaml.SafeLoader)

            if self.cfg == cache_cfg:
                logger.info("Cache verified. Loading...")
                return True
            else:
                logger.info("Cache does not match spec. Rebuilding data.")
                return False
        else:
            logger.info("Cache does not exists, building data.")
            return False

    def save_cache(self, data: xr.Dataset):

        root, ext = os.path.splitext(self.cfg["cache_path"])

        # dump cfg
        # self.cfg = convert_pathlib_opts_to_str(self.cfg)
        yaml.dump(self.cfg, open(root + ".yaml", "w"))

        # dump data
        data.to_netcdf(root + ".nc")

        return True

    def load_cache(self):

        data = xr.load_dataset(self.cfg["cache_path"])

        return data

    def build_dataset(self) -> Dataset:
        # if yes -> build
        if self.cfg["cache_path"] is not None:
            logger.info(f'Checking cache at {self.cfg["cache_path"]}')
            if self.check_cache():
                data = self.load_cache()
            else:
                data = self._build_data()

                # then save path
                self.save_cache(data)

        else:
            logger.info("No cache_path set, building data")
            # cache is null, just build dataset
            data = self._build_data()

        # build datatset now
        ptdataset = self._build_ptdataset(data)

        return ptdataset

    def _build_data(self, merge: bool = True):

        sdt = datetime.strptime(self.cfg["start_data_date"], "%Y-%m-%d")
        edt = datetime.strptime(self.cfg["end_data_date"], "%Y-%m-%d")

        arrays = []

        # data_unit_options: Dict[str, Any]
        for data_unit_name, data_unit_options in self.cfg["data_units"].items():
            data_unit_instance = locate(data_unit_options["class"])()
            array = data_unit_instance.build(
                start_datetime=sdt,
                end_datetime=edt,
                site_mapper=self.get_site_mapper(
                    data_unit_options["site_keys"], self.cfg["sites"]
                ),
                data_unit_name=data_unit_name,
                **data_unit_options,
            )
            arrays.append(array)

        if merge:
            return xr.merge(arrays)
        else:
            return arrays

    def _build_ptdataset(
        self,
        data: xr.Dataset,
    ) -> Dataset:

        PTDataset = locate(self.ptds_cfg["pytorch_dataset"])

        ptdataset = PTDataset(data, **self.ptds_cfg)

        return ptdataset
