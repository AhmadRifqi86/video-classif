FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_STAGE=prod
ENV MODEL_PATH=/app/models/seq60_batch32_hidden32_cnnresnet50_rnninput8_layer3_typemamba_acc0.7842_unidir.pth
ENV SAMPLING_METHOD=uniform
ENV SEQUENCE_LENGTH=60
ENV TZ=Asia/Jakarta

RUN apt-get update -y && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-opencv \
    xvfb \
    wget \
    gnupg \
    curl \
    firefox \
    libdbus-1-3 \
    libx11-xcb1 \
    libxt6 \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxfixes3 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install latest Geckodriver
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz \
 && tar -xvzf geckodriver-v0.30.0-linux64.tar.gz \
 && mv geckodriver /usr/local/bin/ \
 && rm geckodriver-v0.30.0-linux64.tar.gz

# Install Python dependencies
RUN pip install --update
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    torchvision==0.19.1 \
    torchtext==0.18.0 \
    h5py==3.12.1 \
    requests==2.32.3 \
    numpy==1.24.4 \
    scikit-image==0.24.0 \
    einops==0.8.0 \
    TikTokApi==6.5.2 \
    browser_cookie3==0.19.1 \
    pandas==2.2.2 \
    pyzmq==26.2.0 \
    beautifulsoup4==4.12.3 \
    playwright==1.37.0

# Playwright setup
RUN playwright install-deps && \
    playwright install firefox && \
    firefox --headless & (sleep 5 && kill $!) || true

# Copy application files
COPY skripsi/custom_pyktok /app/custom_pyktok
COPY skripsi/medsos_lrcn/src/worker.py /app
COPY skripsi/medsos_lrcn/src/all_config.py /app
COPY skripsi/medsos_lrcn/src/loader_data.py /app
COPY skripsi/medsos_lrcn/src/models_bidir.py /app
COPY skripsi/medsos_lrcn/src/testcv.py /app

CMD ["python3","worker.py"]