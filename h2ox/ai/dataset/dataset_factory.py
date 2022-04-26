import json
import os
from datetime import datetime
from pydoc import locate
from typing import Any, Dict

import xarray as xr
import yaml
from loguru import logger
from torch.utils.data import Dataset


def maybe_load(path):
    if isinstance(path, str):
        if os.path.splitext(path)[-1] == ".json":
            return json.load(open(path))
        elif os.path.splitext(path)[-1] == ".yaml":
            return yaml.load(open(path), Loader=yaml.SafeLoader)
        else:
            raise NotImplementedError
    else:
        return path


class DatasetFactory:
    def __init__(
        self,
        cfg: Dict[str, Any],
    ):
        self.cfg = cfg["data_parameters"]
        self.ptds_cfg = cfg["dataset_parameters"]

        self.sites = maybe_load(self.cfg["sites"])

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

            site_keys = maybe_load(data_unit_options["site_keys"])

            data_unit_instance = locate(data_unit_options["class"])()
            array = data_unit_instance.build(
                start_datetime=sdt,
                end_datetime=edt,
                site_mapper=self.get_site_mapper(site_keys, self.sites),
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

        select_sites = maybe_load(self.ptds_cfg["select_sites"])

        sites_edges = maybe_load(self.ptds_cfg["sites_edges"])

        cfg_copy = {
            kk: vv
            for kk, vv in self.ptds_cfg.items()
            if kk not in ["select_sites", "sites_edges"]
        }

        ptdataset = PTDataset(
            data, select_sites=select_sites, sites_edges=sites_edges, **cfg_copy
        )

        return ptdataset
