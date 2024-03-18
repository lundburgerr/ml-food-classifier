FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
LABEL authors="lundburgerr"

RUN conda install -y tensorboard

RUN conda install -y torchvision>0.16.0

RUN conda install -y torchserve==0.10.0 torch-model-archiver==0.10.0 -c pytorch