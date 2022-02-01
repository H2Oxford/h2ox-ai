import os
from pathlib import Path
from datetime import datetime

import yaml
from loguru import logger
from sacred import Experiment
from sacred.observers import FileStorageObserver, GoogleCloudStorageObserver


CONFIG_PATH = Path.cwd() / "conf.yaml"
GCP_CREDENTIALS_PATH = Path.cwd() / "gcp_credentials.json"
GCP_CONFIG_PATH = Path.cwd() / "gcp_config.yaml"


NAME = "h2ox-ai_" + datetime.now().isoformat()[0:16]
ex = Experiment(NAME)

logger.info(f"Experiment created with {NAME=}")
ex.observers.append(FileStorageObserver("experiments"))
logger.info("Added Observed at ./experiments/")


if GCP_CREDENTIALS_PATH.exists() and GCP_CONFIG_PATH.exists():
    gcp_config = yaml.load(GCP_CONFIG_PATH, Loader=yaml.SafeLoader)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
    ex.observers.append(
        GoogleCloudStorageObserver(
            bucket=gcp_config["bucket"], basedir=gcp_config["basedir"]
        )
    )
    logger.info(
        f"Added GCS Observer at gs://{gcp_config['bucket']}/{gcp_config['basedir']}"
    )
else:
    logger.info("No GCP configuration or credentials found. Omitting GCS observer.")

ex.add_config(CONFIG_PATH.as_posix())
logger.info(f"Added {CONFIG_PATH.as_posix()=}")
