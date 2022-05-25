[<img alt="Wave2Web Hack" width="1000px" src="https://github.com/H2Oxford/.github/raw/main/profile/img/wave2web-banner.png" />](https://www.wricitiesindia.org/content/wave2web-hack)

H2Ox is a team of Oxford University PhD students and researchers who won first prize in the[Wave2Web Hackathon](https://www.wricitiesindia.org/content/wave2web-hack), September 2021, organised by the World Resources Institute and sponsored by Microsoft and Blackrock. In the Wave2Web hackathon, teams competed to predict reservoir levels in four reservoirs in the Kaveri basin West of Bangaluru: Kabini, Krishnaraja Sagar, Harangi, and Hemavathy. H2Ox used sequence-to-sequence models with meterological and forecast forcing data to predict reservoir levels up to 90 days in the future.

The H2Ox dashboard can be found at [https://h2ox.org](https://h2ox.org). The data API can be accessed at [https://api.h2ox.org](https://api.h2ox.org/docs#/). All code and repos can be [https://github.com/H2Oxford](https://github.com/H2Oxford). Our Prototype Submission Slides are [here](https://docs.google.com/presentation/d/1J_lmFu8TTejnipl-l8bXUZdKioVseRB4tTzqK6sEokI/edit?usp=sharing). The H2Ox team is [Lucas Kruitwagen](https://github.com/Lkruitwagen), [Chris Arderne](https://github.com/carderne), [Tommy Lees](https://github.com/tommylees112), and [Lisa Thalheimer](https://github.com/geoliz).


# H2Ox - AI
This repo is for training and deploying the winning models of the h2ox team for the wave2web hackathon.
In total 11 models were trained, for 10 different basin networks, plus one 'extras' with singleton reservoirs.


## Installation

This repo can be `pip` installed:

    pip install https://github.com/H2Oxford/h2ox-ai.git

For development, the repo can be pip installed with the `-e` flag and `[dev]` options:

    git clone https://github.com/H2Oxford/h2ox-ai.git
    cd h2ox-ai
    pip install -e .[dev]

Don't forget to install the `pre-commit` configuration for nice tidy code!

    pre-commit install

For containerised deployment, a docker container can be built from this repo:

    docker build -t <my-tag> .

Cloudbuild container registery services can also be targeted at forks of this repository.


## Description

### Machine Learning Problem

The machine learning problem being approached in this reservoir is the prediction of reservoir volume changes up to 90 days in the future for sixy-six reservoirs across ten river basin networks in India.

The complication is that reservoir volume changes are autocorrelative and networks: reservoirs cannot be filled past a certain volume; and may be filled or emptied depending on the volume available in upstream or downstream basins. The machine learning model needs to be able to capture this context.

**context: ten basin networks and hydrological adjacency**
<img alt="Wave2Web Hack" width="600px" src="https://github.com/H2Oxford/.github/raw/main/profile/img/context.png" />

The primary drivers of water flow-runoff models are meteorological data, primarily precipitation and evapotranspiration.
In this problem, we have collected precipitation and temperature (as a proxy for evapotranspiration) data for the unique upstream area of every reservoir.
We also have hydrological adjacency between reservoirs based on the upstream-downstream relationships.
Can we predict reservoir volume changes up to 90 days in the future to help planners avoid water shortage events?

### Experiment Control and Entrypoint

Training is managed using the [Sacred](https://sacred.readthedocs.io/en/stable/) experiment framework.
The default configuration is located at [conf.yaml](conf.yaml).
Changes made to the configuration will be logger in the sacred experiment directory: `experiments/sacred/{_run._id}`.

The main entrypoint for model training is at [run.py](run.py), which can just be invoked from the commandline:

    python run.py

### Data

This repo contains everything needed to serve the h2ox models.
To retrain the models, a `DatasetFactory` is provided to rebuild the data archives for each basin network.
The final configuration `.yaml` files are also provided.
The data archives used to train the basin networks are available at [gs://oxeo-public/wave2web/h2ox-ai](https://console.cloud.google.com/storage/browser/oxeo-public/wave2web/h2ox-ai).
Full experiment logs and artefacts can be provided on request.

#### Dataset Factory

An abstract `DatasetFactory` class is used for caching and version controlling datasets used to train various basin networks.
The `DatasetFactory` class builds a NetCDF4 `.nc` data archive that can be used as the input for the `FcastDataset`, an subclass of a PyTorch Dataset.
This `.nc` archive is accompanied by a `data.yaml` file which is used to certify version control of the data archive.
In this way, the expensive dataset construction steps are able to be cached, and soft data rules and hyperparameters are left to the `Dataset` construction at runtime. The `DatasetFactory` uses the `data_parameters` field of the experiment `conf.yaml` to contruct the data archive and verify version control.

Custom `DataUnits` are passed to the `DatasetFactory`.
These can be found in [DataUnits](h2ox/ai/dataset/data_units.py) and allow data archives of varying parameters to be built and cached.
`DataUnits` contain the logic for accessing, for example, [zarr](https://zarr.readthedocs.io/en/stable/) archives build in `h2ox-data`,`h2ox-chirps`, and `h2ox-forecast`, as well as BigQuery tables built in `h2ox-w2w`.
Data archives include [ECMWF-ERA5Land historical data](https://www.ecmwf.int/en/era5-land), [ECMWF-TIGGE ex-ante forecast data](https://www.ecmwf.int/en/research/projects/tigge), [CHIRPS precipitation data](https://www.chc.ucsb.edu/data/chirps), [WRIS Reservoir Volumes](https://indiawris.gov.in/wris/), and day-of-the-year trigonometric data.

**data: sample dataframe (Indira Sagar)**
<img alt="Wave2Web Hack" width="800px" src="https://github.com/H2Oxford/.github/raw/main/profile/img/indira_sample.png" />

#### Pretrained Models

As the models are relatively small, they are shipped with this repository.
They can be found in [/models](models/), both the `.pt` files, the `.mar` archives, along with accompanying config files, dummy items, reservoir lists, edge lists, and preprocessing parameters (means, standard deviations etc.) for de-normalising reservoir volume changes.

### Model

H2Ox developed a Sequence-to-sequence-to-sequence LSTM model: a three-stage LSTM which 1) encodes historic meteorological forcing data into a latent hydrological state for each reservoir; 2) decodes over a forecast period, while continuing conditioning from forecast data; and 3) decodes further over a future period, with only trigonometic day-of-the-year features as input.

**model: bayesian seq2seq2seq LSTM with graph conv header**
<img alt="Wave2Web Hack" width="1000px" src="https://github.com/H2Oxford/.github/raw/main/profile/img/w2w_model.png" />

Several configurations are available for training this model.
Model configuration is controlled by the `model_parameters` field of the `conf.yaml` file.


#### Baseline

A baseline configuration has the following `model_parameters`:

    model_str: s2s2s
    graph_conv: false
    bayesian_lstm: false

Two variations are available: one-hot-encoding (**ohe**) or multi-site encoding (**multi**). One-hot-encoding concatenates input data to be indexed by `<date>-<reservoir>` and adding an input feature to flag which reservoir is the target. The intention of this variation is to make the training dataframe larger and learn generaliseable information about the hydrological system. A multi-target configuration is designed to allow the model to learn parameterisation of how each reservoir in the dataset impacts each other reservoir (without injecting heuristical knowledge of how this might be so).

The variation is chosen by setting `dataset_parameters.ohe_or_multi` to the respective either `ohe` or `multi`.

#### Bayesian

A bayesian configuration has been added which makes encoder and decoder weights probabilistic, sampled from a normal distribution.
This configuration uses the [blitz](https://github.com/piEsposito/blitz-bayesian-deep-learning) bayesian layer implementations.
To enable the bayesian configuration, change the `model_parameters` to:

    model_str: bayesian
    graph_conv: false
    bayesian_lstm: true

The bayesian configuration makes use of the following hyperparameters which control starting assumptions of the distribution of model weight sampling:

    lstm_params:
        prior_pi: 1.
        prior_sigma_1: 0.1
        prior_sigma_2: 0.002
        posterior_mu_init: 0
        posterior_rho_init: -3

`dataset_parameters.ohe_or_multi` should be set to `sitewise`.


#### GNN

The final configuration adds a graph neural network (GNN) header on top of the bayesian LSTM encoder and decoders.
The graph convolutional layer traverses information between the respective reservoirs in a basin using the hydrological adjacency graph.
To enable this configuration, set `model_parameters` to:

    model_str: gnn
    graph_conv: true
    bayesian_lstm: true

The parameters `diag` and `digraph` control properties of the adjacency matrix. `diag` add ones along the diagonal of the adjacency graph. `digraph` adds ones to the opposite quadrant of the adjacency matrix.


### Sample Data

A sample cached dataset `test_data.nc` and config record `test_data.yaml` are available at [gs://oxeo-public/wave2web/h2ox-ai](https://console.cloud.google.com/storage/browser/oxeo-public/wave2web/h2ox-ai).
Final cached datasets for training each basin network are also available.
The whole data folder can be copied to a local `data/` folder using the `gsutil` command line utility:

    gsutil -m cp gs://oxeo-public/wave2web/h2ox-ai/data/* ./data

This data can be copied to `data/` to begin experimenting with the repository without regenerating the full dataset.
Remember to change the cache path in both `conf.yaml` and `data/test_data.yaml`!
`basins.geojson` and `all_edges.json` must also be copied to the data folder.

## Useage

### Training

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


### Serving

This repo is also set up for production as a dockerised [TorchServe](https://pytorch.org/serve/) instance.
[serve.py](serve.py) provides a custom TorchServe handler to receive, preprocess, predict, and postprocess requests.

To serve any new or retrained models,

    torch-model-archiver --model-name kaveri --version 0.1 --serialized-file models/kaveri.pt --extra-files models/kaveri_preprocessing.json,models/all_edges.json,models/kaveri_sites.json,models/kaveri_cfg.json,models/kaveri_dummy_item.pkl --handler /home/lucas/h2ox-ai/serve.py

To serve a specific model:

    torchserve --model-store models/ --start --ncs --models kaveri=kaveri.mar

To serve all modes:

    torchserve --model-store models/ --start --ncs --models all

A sample can then be requested:

    import requests, json

    headers = {"Content-Type": "application/json"}
    sample = {
        'instances: [json.load(open('./data/kaveri_sample_2020_10_01.json','r'))]
    }
    url = 'http://0.0.0.0:7080/predictions/kaveri'

    r = requests.post(url,headers=headers, data=json.dumps(sample).encode('utf-8'))

    print ('status code:', r.status_code)


## Citation


Our Wave2Web submission can be cited as:

    <citation here>
