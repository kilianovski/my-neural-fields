FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG PIP_EXTRA_INDEX_URL="https://pypi.org/simple/"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \ 
    libsm6 \
    libxext6 \
    ffmpeg \
    libxrender-dev \
    zip \
    unzip

RUN apt-get install build-essential -y
RUN apt-get install git -y
RUN git config --global --add safe.directory /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

RUN python -m pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt200/download.html

RUN pip install Cython
