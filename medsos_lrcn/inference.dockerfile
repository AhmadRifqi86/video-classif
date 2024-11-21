# Use a base image with PyTorch and CUDA 12.2 support
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install Python and other required dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python script, model, and configuration files
COPY deployment.py .
COPY all_config.py .
COPY loader_data.py .
# nanti bikin requirements.txt untuk inference
COPY inference_requirements.txt .  

# Copy the model file into the container
COPY models/best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth /app/models/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose any necessary ports if the application has HTTP endpoints (if not needed, remove this line)
EXPOSE 5000

# Define the entrypoint to allow passing CLI arguments
ENTRYPOINT ["python3", "deployment.py"]

# Set a default command (can be overridden when running the container)
CMD ["--model", "best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth"]



#build
#docker build -t [dockerfile name] .

#run
#docker run --rm --gpus all -v [VIDEO_DIR]:/app/videos video-classifier-cuda12 --model best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth

