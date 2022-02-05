import os
from datetime import datetime
from pydoc import locate

import xarray as xr
import yaml
from loguru import logger
from torch.utils.data import Dataset


class DatasetFactory:
    def __init__(
        self,
        cfg,
    ):
        pass

    @staticmethod
    def get_site_mapper(data_unit_site_keys, global_site_keys):
        """a placeholder method just to map site keys for each DataUnit to global site keys"""
        return dict(zip(data_unit_site_keys, global_site_keys))

    def check_cache(self):

        # if exists
        if os.path.exists(self.cfg["cache_path"]):
            # load cfg
            root, ext = os.path.splitext(self.cfg["cache_path"])

            cache_cfg = yaml.load(
                open(os.path.join(root, ".yaml")), Loader=yaml.SafeLoader
            )

            if self.cfg == cache_cfg:
                logger.info("Cache verified. Loading...")
                return True
            else:
                logger.info("Cache doesnt match spec. Rebuilding data.")
                return False
        else:
            logger.info("Cache does not exists, building data.")
            return False

    def save_cache(self, data: xr.Dataset):

        root, ext = os.path.splitext(self.cfg["cache_path"])

        yaml.dump(self.cfg, open(os.path.join(root, ".yaml"), "w"))

        return True

    def load_cache(self):

        data = xr.load_dataset(self.cfg["cache_path"])

        return data

    def build_dataset(self):

        # need to build?

        # if yes -> buil
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

    def _build_data(self):

        sdt = datetime.strptime(self.cfg["end-datetime"], "%Y-%m-%d")
        edt = datetime.strptime(self.cfg["start-datetime"], "%Y-%m-%d")

        arrays = []

        for kk, vv in self.cfg["data_units"].items():
            data_unit_instance = locate(vv["class"])()
            array = data_unit_instance.build(
                start_datetime=sdt,
                end_datetime=edt,
                site_mapper=self.get_site_mapper(vv["site_keys"], self.cfg["sites"]),
                data_unit_name=kk,
                **vv,
            )
            arrays.append(array)

        return xr.merge(arrays)

    def _build_ptdataset(
        self,
        data: xr.Dataset,
    ) -> Dataset:

        return data
