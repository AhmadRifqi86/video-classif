FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV APP_STAGE=prod
ENV MODEL_PATH=/app/models/seq60_batch32_hidden32_cnnresnet50_rnninput8_layer3_typemamba_acc0.7842_unidir.pth
ENV SAMPLING_METHOD=uniform
ENV SEQUENCE_LENGTH=60
ENV VIDEO_DIR=/tmp/tmpfs_video
# Set the timezone to avoid timezone prompts
ENV TZ=Asia/Jakarta

RUN mkdir -p $VIDEO_DIR

RUN apt-get update -y
RUN apt-get install python3 python3-pip -y
# Install system dependencies required for PyTorch, OpenCV, and Python
RUN apt-get update -y
RUN apt-get install python3-opencv -y && apt-get clean && rm -rf /var/lib/apt/lists/*
# Install Firefox
RUN apt-get update -y
RUN apt-get install firefox-esr wget gnupg curl unzip

# Configure timezone to avoid prompts
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

#RUN pip install --upgrade pip
RUN pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 torchtext==0.18.0 
RUN pip install h5py requests==2.32.3 numpy==1.24.4 scikit-image einops
RUN pip install TikTokApi==6.5.2 browser_cookie3==0.19.1 pika

# Copy the application code into the container
COPY skripsi/custom_pyktok /app/custom_pyktok
COPY skripsi/medsos_lrcn/src/deployment.py /app
COPY skripsi/medsos_lrcn/src/all_config.py /app
COPY skripsi/medsos_lrcn/src/loader_data.py /app
COPY skripsi/medsos_lrcn/src/models_bidir.py /app
COPY skripsi/medsos_lrcn/src/testcv.py /app

RUN firefox --headless & (sleep 5 && kill $!) || true