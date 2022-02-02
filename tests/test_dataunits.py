from datetime import datetime, timedelta
from pydoc import locate

import yaml


def test(sdt, edt, cfg):

    for kk, vv in cfg["data_units"].items():
        ob = locate(vv["class"])
        inst = ob()
        array = inst.build(
            start_datetime=sdt,
            end_datetime=edt,
            site_keys=dict(zip(vv["sites"], cfg["sites"])),
            data_unit_name=kk,
            **vv
        )


if __name__ == "__main__":
    cfg = yaml.load(open("./tests/test.yaml"), Loader=yaml.SafeLoader)

    sdt = (
        min(
            datetime.strptime(cfg["train_start_date"], "%Y-%m-%d"),
            datetime.strptime(cfg["test_start_date"], "%Y-%m-%d"),
        )
        - timedelta(days=cfg["seq_len"])
    )
    edt = (
        max(
            datetime.strptime(cfg["train_start_date"], "%Y-%m-%d"),
            datetime.strptime(cfg["test_start_date"], "%Y-%m-%d"),
        )
        + timedelta(days=cfg["future_horizon"])
    )

    test(sdt, edt, cfg)
