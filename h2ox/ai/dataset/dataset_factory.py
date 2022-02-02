from pydoc import locate

from loguru import logger
from torch.utils.data import Dataset

from h2ox.ai.dataset.data_units import DataUnit
from h2ox.ai.dataset.dataset import FcastDataset


class DatasetFactory:

    cfg
    build_dataset
    save_cache
    check_cache
    load_cache



    def check_cache():

    def save_cache():

    def load_cache():

    def build_dataset():


        return ptdataset

    def _build_array():

        pass



    def _build_dataset(
        self,
        data_units: Sequence[DataUnit],
        gdf: gpd.GeoDataFrame,
        sites: List[str],
    ) -> Dataset:

        return ptdataset


if __name__=="__main__":

    cfg =
