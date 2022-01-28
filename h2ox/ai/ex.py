import os
from datetime import datetime

import yaml
from loguru import logger
from sacred import Experiment
from sacred.observers import FileStorageObserver, GoogleCloudStorageObserver

CONFIG_PATH = os.path.join(os.getcwd(), "conf.yaml")
GCP_CREDENTIALS_PATH = os.path.join(os.getcwd(), "gcp_credentials.json")
GCP_CONFIG_PATH = os.path.join(os.getcwd(), "gcp_config.yaml")


NAME = "h2ox-ai_" + datetime.now().isoformat()[0:16]
ex = Experiment(NAME)

logger.info(f"Experiment created with {NAME=}")
ex.observers.append(FileStorageObserver("experiments"))
logger.info("Added Observed at /experiments/")


if os.path.exists(GCP_CREDENTIALS_PATH) and os.path.exists(GCP_CONFIG_PATH):
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

ex.add_config(CONFIG_PATH)
logger.info(f"Added {CONFIG_PATH=}")
