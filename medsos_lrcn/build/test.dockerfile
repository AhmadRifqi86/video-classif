FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set the working directory
WORKDIR /app

# Copy the Python script, model, and configuration files
COPY skripsi/medsos_lrcn/src/testcv.py /app/

# Define the entrypoint to allow passing CLI arguments
CMD ["python3", "testcv.py"]

# Set a default command (can be overridden when running the container)
#CMD ["--model", "/app/models/best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth"]

#docker build -f skripsi/medsos_lrcn/build/test.dockerfile -t testcv .
#docker run -it --rm -v /home/arifadh/Downloads/tiktok_videos:/app testcv


#docker run --rm -v /home/arifadh/Downloads/tiktok_videos:/app/videos testcv bash
