# Use a base image with PyTorch and CUDA 12.2 support
#FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set the working directory
WORKDIR /app

# Install Python and other required dependencies

# Copy the Python script, model, and configuration files
COPY skripsi/medsos_lrcn/src/deployment.py .
COPY skripsi/medsos_lrcn/src/all_config.py .
COPY skripsi/medsos_lrcn/src/loader_data.py .
COPY skripsi/medsos_lrcn/src/models.py .
# nanti bikin requirements.txt untuk inference
COPY skripsi/medsos_lrcn/build/inference_req.txt .  

# Copy the model file into the container
#COPY best_models_medsos/best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth /app/models/

# Install Python dependencies
RUN pip install --no-cache-dir -r ./inference_req.txt

# Expose any necessary ports if the application has HTTP endpoints (if not needed, remove this line)
#EXPOSE 5000

# Define the entrypoint to allow passing CLI arguments
ENTRYPOINT ["python3", "deployment.py"]

# Set a default command (can be overridden when running the container)
CMD ["--model", "/app/models/best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth"]



#build
#cd ~/Desktop/Skripsi-Magang-Proyek
#docker build -f inference.dockerfile -t inference .

#run
#docker run --rm --gpus all -v /home/arifadh/:/app/videos video-classifier-cuda12 --model best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth

