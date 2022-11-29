FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN mkdir -p /home/ubuntu
WORKDIR /home/ubuntu
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.7 python3.7-dev python3-pip python3-setuptools python3-wheel gcc
RUN apt-get install -y git
RUN python3.7 -m pip install pip --upgrade

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --requirement requirements.txt


