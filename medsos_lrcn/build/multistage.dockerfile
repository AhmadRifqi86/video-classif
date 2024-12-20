# Stage 1: Build Stage
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as build-stage
WORKDIR /app

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jakarta

# Install required system dependencies
RUN apt-get update -y && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    software-properties-common \
    wget \
    curl \
    gnupg \
    unzip

# Add Mozilla PPA and configure Firefox pinning
RUN add-apt-repository -y ppa:mozillateam/ppa && \
    echo 'Package: * \nPin: release o=LP-PPA-mozillateam \nPin-Priority: 1001 \n\nPackage: firefox\nPin: version 1:1snap1-0ubuntu2\nPin-Priority: -1\n' | tee /etc/apt/preferences.d/mozilla-firefox

# Configure timezone
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install Firefox and Geckodriver
RUN apt-get update -y && apt-get install -y firefox && rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz \
    && tar -xvzf geckodriver-v0.30.0-linux64.tar.gz \
    && mv geckodriver /usr/local/bin/ \
    && rm geckodriver-v0.30.0-linux64.tar.gz

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    torchvision==0.19.1 \
    torchtext==0.18.0 \
    h5py==3.12.1 \
    requests==2.32.3 \
    scikit-image==0.24.0 \
    numpy==1.26.4 \
    einops==0.8.0 \
    browser_cookie3==0.19.1 \
    pandas==2.2.2 \
    pyzmq==26.2.0 \
    beautifulsoup4==4.12.3

# Copy application files
COPY skripsi/custom_pyktok /app/custom_pyktok
COPY skripsi/medsos_lrcn/src/worker.py /app
COPY skripsi/medsos_lrcn/src/all_config.py /app
COPY skripsi/medsos_lrcn/src/loader_data.py /app
COPY skripsi/medsos_lrcn/src/models_bidir.py /app
COPY skripsi/medsos_lrcn/src/testcv.py /app

# Validate Firefox
RUN firefox --headless & (sleep 5 && kill $!) || true

# Stage 2: Final Stage
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
WORKDIR /app

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jakarta


# Copy system dependencies from build stage
COPY --from=build-stage /usr/local/bin/geckodriver /usr/local/bin/geckodriver
COPY --from=build-stage /usr/lib/firefox /usr/lib/firefox
COPY --from=build-stage /etc/apt /etc/apt

# Configure timezone
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install Python and necessary dependencies
RUN apt-get update -y && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build-stage /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages

# Copy application files from the build stage
COPY --from=build-stage /app /app

ENV VIDEO_DIR=/app/videos
ENV MODEL_PATH=/app/models/seq60_max1000_mobilenetv2_mamba_batch32_hidden32_rnninp8_layer3_unidir_adp123normsiludrop0.3_dinner4dmodel_dtrankeqnstate_epoch8_acc0.8189.pth
ENV SAMPLING_METHOD=uniform
ENV SEQUENCE_LENGTH=60

RUN mkdir -p $VIDEO_DIR

EXPOSE 54000