FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV APP_STAGE=prod
ENV VIDEO_DIR=/app/videos
ENV MODEL_PATH=/app/models/seq60_batch32_hidden32_cnnresnet50_rnninput8_layer3_typemamba_acc0.7842_unidir.pth
ENV SAMPLING_METHOD=uniform
ENV SEQUENCE_LENGTH=60
ENV TZ=Asia/Jakarta

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

RUN add-apt-repository -y ppa:mozillateam/ppa

RUN echo 'Package: * \nPin: release o=LP-PPA-mozillateam \nPin-Priority: 1001 \n\nPackage: firefox\nPin: version 1:1snap1-0ubuntu2\nPin-Priority: -1\n' | tee /etc/apt/preferences.d/mozilla-firefox

RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update -y && apt-get install -y firefox && rm -rf /var/lib/apt/lists/*
# Install latest Geckodriver
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

# Playwright setup
RUN firefox --headless & (sleep 5 && kill $!) || true
RUN mkdir -p $VIDEO_DIR

EXPOSE 54000

CMD ["python3","worker.py"]

#docker run --gpus all -d --name worker --network backend-network -p 54000:54000 -v /home/arifadh/Desktop/Skripsi-Magang-Proyek/best_models_medsos2:/app/models worker python3 worker.py




