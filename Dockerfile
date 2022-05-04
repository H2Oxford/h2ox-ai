# Dockerfile for VertexAI endpoint
FROM pytorch/torchserve:0.5.3-cpu

# install dependencies

RUN echo $(python --version)
RUN echo $(which pip)


# copy model artifacts, custom handler and other dependencies
COPY . /home/model-server/h2ox
COPY ./models /home/model-server/models

#RUN pip install setuptools

USER root
RUN pip install /home/model-server/h2ox

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

# create model archive file packaging model artifacts and dependencies
# RUN torch-model-archiver -f \
#  --model-name=$APP_NAME \
#  --version=1.0 \
#  --serialized-file=/home/model-server/pytorch_model.bin \
#  --handler=/home/model-server/custom_text_handler.py \
#  --extra-files "/home/model-server/config.json,/home/model-server/tokenizer.json,/home/model-server/training_args.bin,/home/model-#server/tokenizer_config.json,/home/model-server/special_tokens_map.json,/home/model-server/vocab.txt,/home/model-server/index_to_name.json" \
#  --export-path=/home/model-server/model-store

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "all", \
     "--model-store", \
     "/home/model-server/models"]
