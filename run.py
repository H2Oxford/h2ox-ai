import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from h2ox.ai.dataset import FcastDataset
from h2ox.ai.train import train, test
# from h2ox.ai.utils import normalize_data, unnormalize_preds
from h2ox.ai.dataset import FcastDataset
from h2ox.ai.model import initialise_model
from h2ox.scripts.utils import load_zscore_data
from definitions import ROOT_DIR


# TODO: normalize/unnormalize data
# TODO: save experiments in reproducible way
# TODO: test dataloader 

if __name__ == "__main__":
    # parameters for the yaml file
    ENCODE_DOY = True
    SEQ_LEN = 60
    FUTURE_HORIZON = 76
    SITE = "kabini"
    TARGET_VAR = "volume_bcm"
    HISTORY_VARIABLES = ["tp", "t2m"]
    FORECAST_VARIABLES = ["tp", "t2m"]
    BATCH_SIZE = 32
    TRAIN_END_DATE = "2018-12-31"
    TRAIN_START_DATE = "2010-01-01"
    TEST_START_DATE = "2019-01-01"
    TEST_END_DATE = "2022-01-01"
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    DROPOUT = 0.4
    NUM_WORKERS = 4
    RANDOM_VAL_SPLIT = True
    EVAL_TEST = True
    N_EPOCHS = 30
