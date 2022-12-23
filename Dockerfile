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

WORKDIR $APP_HOME


CMD ["python", "batch.py"]
