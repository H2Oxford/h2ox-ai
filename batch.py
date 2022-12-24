import gc
import json
import os
from datetime import datetime

from loguru import logger

from h2ox.ai.inference import local_inference
from h2ox.ai.slackbot import SlackMessenger


def main():

    today = datetime.now()

    token = os.environ.get("SLACKBOT_TOKEN")
    target = os.environ.get("SLACKBOT_TARGET")

    if token is not None and target is not None:

        slackmessenger = SlackMessenger(
            token=token,
            target=target,
            name="H2OX-PIPELINE :: ",
        )
    else:
        slackmessenger = None

    msg, code = w2w_inference(today, slackmessenger)
    logger.info("made it here")

    return f"Done inference {today.isoformat()}", 200


def w2w_inference(today: datetime, slackmessenger: SlackMessenger):

    # step 2-> rerun inference and post results
    basin_networks = json.loads(os.environ.get("BASIN_NETWORKS"))
    msg = local_inference(today, basin_networks)
    if slackmessenger is not None:
        slackmessenger.message(f"W2W ::: inference: {json.dumps(msg)}")

    return f"w2w inference {today.isoformat()}", 200


if __name__ == "__main__":
    main()
    logger.info("made it there")
    gc.collect()
