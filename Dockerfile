# Dockerfile for VertexAI endpoint
FROM pytorch/torchserve:latest-cpu
# install dependencies


# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True


ENV APP_HOME /app

COPY ./batch.py $APP_HOME/

# Copy local code to the container image.
# __context__ to __workdir__
COPY . h2ox-ai

USER root
RUN pip install h2ox-ai/[docker]

# create torchserve configuration file
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\nnumber_of_gpu=0" >> /home/model-server/config.properties
RUN printf "\ndefault_workers_per_model=1" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties

USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

WORKDIR $APP_HOME

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "all", \
     "--model-store", \
     "/home/model-server/h2ox-ai/models"]
