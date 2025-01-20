FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG USERNAME
ARG UID
ARG USERHOME

RUN mkdir -p ${USERHOME}
RUN useradd -m -d ${USERHOME} -u ${UID} ${USERNAME}
RUN chown -R ${UID}:${UID} ${USERHOME}

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
RUN pip install pybind11 

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY /hf_file /hf_file
RUN chown ${UID}:${UID} /hf_file

USER ${USERNAME}
RUN huggingface-cli login --token $(cat /hf_file)

USER root

RUN rm /hf_file
