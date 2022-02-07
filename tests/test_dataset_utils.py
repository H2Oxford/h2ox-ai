from pathlib import Path
from h2ox.ai.dataset.utils import load_zscore_data, load_samantha_data, load_reservoir_metas, get_all_big_q_data_as_xarray


if __name__ == "__main__":
    data_dir = Path(Path.cwd() / "data")
    target, history, forecast = load_zscore_data(data_dir)
    sam_data, meta = load_samantha_data(data_dir)
    bigq_meta = load_reservoir_metas(data_dir)
    bigq_target = get_all_big_q_data_as_xarray(data_dir)