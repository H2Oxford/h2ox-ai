# h2ox-ai
This repo is for training machine learning models for the Wave2Web hackathon.

## Description

[<img alt="Wave2Web Hack" width="800px" src="docs/img/wave2web-banner.png" />](https://www.wricitiesindia.org/content/wave2web-hack)

The Wave2Web hackathon, organised by the World Resources Institute and sponsored by Microsoft and Blackrock, took place May through September 2021.
The objective of the hackathon was to predict, up to 90 days in the future, the water availability at 4 key reservoirs in the Kaveri basin to the West of Bengaluru.

<img alt="Kaveri Basin" width="800px" src="docs/img/kaveri-w2w.png" />

The machine learning approach taken by the winning team, H2Ox, is developed in this repository.

### Data

This repo contains everything needed to retrain the winning model of the hackathon.
The primary data used to train the models includes historical forcing data from [ECMWF ERA5-Land Archive](https://www.ecmwf.int/en/era5-land).
Meteorological forcing data for the forecast period is obtained from the [ECMWF TIGGE Global Ensemble Forecast](https://www.ecmwf.int/en/research/projects/tigge).
Reservoir levels are obtained in near-real-time from the [India Water Resources Information System](https://indiawris.gov.in/wris/).

This data automatically assembles the data described in [conf.yaml](conf.py)`/data_parameters`.
Data sources are described in individual [DataUnits](h2ox/ai/dataset/data_units.py) which can be easily added to, to add features to the dataset.
A raw [xarray DataSet](https://xarray.pydata.org/en/stable/generated/xarray.Dataset.html) is cached as a `.nc` file to avoid the data needing to be obtained at the beginning of every training iteration.
In this way, this repo is extensible to new locations, and new data sources, in new applications and use cases.

### Model

H2Ox developed a Sequence-to-sequence-to-sequence LSTM model: a three-stage LSTM which 1) encodes historic meteorological forcing data into a latent hydrological state for each reservoir; 2) decodes over a forecast period, while continuing conditioning from forecast data; and 3) decodes further over a future period, with only trigonometic day-of-the-year features as input.

<img alt="Sequence-to-sequence-to-sequence" width="800px" src="docs/img/s2s2s.png" />
<sub><a href="https://github.com/lkulowski/LSTM_encoder_decoder">Source: L. Kulowski</a></sub>

Several configurations are available for training this model.

#### One-Hot-Encoding

The baseline configuration is a one-hot-encoding of each site.
This allows the model to be used to predict water availability at any given site, trained on mismatched amounts of data by site, and learn general relationships between the meteorological forcing data and the site-wise targets.

#### Multi-Target

To be completed.


## Installation

### For use

To use this repo, simply clone it and then pip install it.

    pip install .

### For Development

    # install with "-e" (editable) and [dev] flag to get pre-commit
    pip install -e .[dev]

    # install pre-commit
    pre-commit install

### Test Data

A sample cached dataset `test_data.nc` and config record `test_data.yaml` are available at [gs://oxeo-public/wave2web/h2ox-ai](https://console.cloud.google.com/storage/browser/oxeo-public/wave2web/h2ox-ai).
This data can be copied to `data/` to begin experimenting with the repository without regenerating the full dataset.
Remember to change the cache path in both `conf.yaml` and `data/test_data.yaml`!
`basins.geojson` describes the upstream areas of each of the 6 test reservoirs, and can be used to regenerate the input data.


## Training

Training is managed using the [Sacred](https://sacred.readthedocs.io/en/stable/) experiment framework.
The default configuration is located at [conf.py](conf.py).
Changes made to the configuration will be logger in the sacred experiment directory: `experiments/sacred/{_run._id}`.

The main entrypoint for model training is at [run.py](run.py), which can just be invoked from the commandline:

    python run.py

Sacred also allows configuration overrides via the commandline.
You can use DotDict notation for nested parameters.
Separate with spaces for multiple options.
For example:

    python run.py with "training_parameters.n_epochs=50" "model_parameters.dropout=0.5"

You can also copy the entire config and make a new version to pass to the experiment object:

    python run.py with my_new_config.yaml

Training writes to a tensorboard `SummaryWriter`.
A tensorboard server can be initiated to monitor your experiments:

    tensorboard --logdir experiments/tensorboard

On your local machine, you can SSH tunnel into the tensorboard server and view your experiment progress in your browser.
By default, tensorboard serves at `\$PORT=6006`.
For example, on GCP:

    gcloud compute ssh --zone "<your-instance-zone>" "<your-instance-name>"  --project "<your-instance-project>" -- -L 6006:localhost:6006

And then view your running tensorboard by directing your browser to `localhost:6006`.


## Serving

This repo is also set up for production as a dockerised [TorchServe](https://pytorch.org/serve/) instance.
The main entrypoint can be found in `app.py`.
(To Be Completed).
