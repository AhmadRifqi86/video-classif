# Use NVIDIA CUDA 12.2 runtime base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set the timezone to avoid timezone prompts
ENV TZ=Asia/Jakarta

RUN apt-get update -y
RUN apt-get install python3 python3-pip -y
# Install system dependencies required for PyTorch, OpenCV, and Python
RUN apt-get update -y
RUN apt-get install python3-opencv -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure timezone to avoid prompts
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

#RUN pip install --upgrade pip
RUN pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 torchtext==0.18.0 
RUN pip install h5py requests==2.32.3 numpy==1.24.4 scikit-image einops

# Copy the application code into the container
COPY skripsi/medsos_lrcn/src/deployment.py /app
COPY skripsi/medsos_lrcn/src/all_config.py /app
COPY skripsi/medsos_lrcn/src/loader_data.py /app
COPY skripsi/medsos_lrcn/src/models.py /app
COPY skripsi/medsos_lrcn/src/testcv.py /app

# Define the command to run your application
CMD ["python3", "deployment.py"]

#cd /home/arifadh/Skripsi-Magang-Proyek
#docker build -f skripsi/medsos_lrcn/build/deployment.dockerfile -t deploy .
#docker run --gpus all -it --rm -v /home/arifadh/Desktop/Skripsi-Magang-Proyek/best_models_medsos:/app/models -v /home/arifadh/Downloads/tiktok_videos:/app/videos deploy python3 deployment.py --model /app/models/best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth --videos /app/videos

