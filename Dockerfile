# Dockerfile for VertexAI endpoint
FROM python:3.9-slim-buster
# install dependencies

# add git & JDK
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install default-jdk

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True


ENV APP_HOME /app

COPY ./batch.py $APP_HOME/

# Copy local code to the container image.
# __context__ to __workdir__
COPY . h2ox-ai

# set up torch serve first
#RUN git clone https://github.com/pytorch/serve.git
#WORKDIR serve
#RUN python ./ts_scripts/install_dependencies.py

# install torch
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
# separate version
RUN pip install torchvision>=0.5

RUN pip install torchserve torch-model-archiver torch-workflow-archiver

# install the rest of the requirements
WORKDIR /
RUN pip install h2ox-ai/[docker]

# create torchserve configuration file
RUN printf "\nservice_envelope=json" >> /app/config.properties
RUN printf "\nnumber_of_gpu=0" >> /app/config.properties
RUN printf "\ndefault_workers_per_model=1" >> /app/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /app/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /app/config.properties

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

WORKDIR $APP_HOME

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=/app/config.properties", \
     "--models", \
     "all", \
     "--model-store", \
     "/h2ox-ai/models"]
