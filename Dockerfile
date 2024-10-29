FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG USERNAME
ARG UID
ARG WANDB_KEY_FILE

RUN useradd -u ${UID} ${USERNAME}

RUN apt update --fix-missing
RUN apt install build-essential -y
RUN apt clean
RUN apt install ffmpeg libsm6 -y
RUN apt clean
RUN apt install neovim -y
RUN apt clean
RUN apt install imagemagick -y
RUN apt clean

RUN pip install --upgrade pip


RUN pip install wandb

COPY ${WANDB_KEY_FILE} /wandb_key
RUN chown ${USERNAME} /wandb_key
RUN su ${USERNAME}
RUN wandb login $(cat /wandb_key)
RUN rm /wandb_key
RUN su


RUN pip install pybind11 

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
