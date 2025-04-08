FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
WORKDIR /app

#System variables
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_STAGE=prod
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
COPY skripsi/medsos_lrcn/src/model2.py /app
COPY skripsi/medsos_lrcn/src/testcv.py /app

# Run firefox for building profile
RUN firefox --headless & (sleep 5 && kill $!) || true

# Model Inference Variables
ENV VIDEO_DIR=/app/videos
ENV MODEL_PATH=/app/models/seq60_batch16_hidden48_cnnresnet50_rnn16_layer3_rnnTypemamba_drop0.4_bidirFalse_acc0.8240_f10.8239.pth
ENV SAMPLING_METHOD=uniform
ENV SEQUENCE_LENGTH=60
RUN mkdir -p $VIDEO_DIR

EXPOSE 54000

CMD ["python3","worker.py"]

#docker run --gpus all -d --name worker --network backend-network -p 54000:54000 -v /home/arifadh/Desktop/Skripsi-Magang-Proyek/grid_best_models_medsos:/app/models worker python3 worker.py

# tambah -e untuk ganti model, sequence_length, sampling method
#docker run --gpus all -it --rm --name worker --network backend-network -p 54000:54000 -v /home/arifadh/Desktop/Skripsi-Magang-Proyek/grid_best_models_medsos:/app/models worker bash
#test script:
#curl -X GET "http://localhost:5000/get_labels?url=https://www.tiktok.com/@devtyyyy/video/7375422441587313926"


#docker run --gpus all -d --name worker --network backend-network -p 54000:54000 -v /home/arifadh/Desktop/Skripsi-Magang-Proyek/grid_best_models_medsos:/app/models -e MODEL_PATH=/app/models/seq60_batch16_hidden48_cnnresnet50_rnn16_layer3_rnnTypelstm_drop0.4_bidirFalse_acc0.7959_f10.7945.pth worker python3 worker.py

#docker run --gpus all -d --name worker --network backend-network -p 54000:54000 -v /home/arifadh/Desktop/Skripsi-Magang-Proyek/grid_best_models_medsos:/app/models -e MODEL_PATH=/app/models/seq60_batch16_hidden64_cnnmobilenet_v2_rnn16_layer3_rnnTypemamba_drop0.5_bidirFalse_acc0.7934_f10.7940.pth worker python3 worker.py

#docker run --gpus all -d --name worker --network backend-network -p 54000:54000 -v /home/arifadh/Desktop/Skripsi-Magang-Proyek/grid_best_models_medsos:/app/models -e MODEL_PATH=/app/models/seq60_batch16_hidden64_cnnmobilenet_v2_rnn16_layer3_rnnTypelstm_drop0.4_bidirFalse_acc0.7883_f10.7888.pth worker python3 worker.py
