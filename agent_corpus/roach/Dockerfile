
# can replace with your base env
# we need python 3.8
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

# Python 3.8 and base system
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip python3.8-dev \
                       wget curl git libglib2.0-0 libsm6 libxext6 libxrender-dev \
                       tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    ln -sf python3.8 /usr/bin/python && \
    python -m pip install --upgrade pip

RUN apt-get update && apt-get install -y libgl1

WORKDIR /app
COPY . /app

RUN bash /app/install_base.sh
