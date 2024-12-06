FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
WORKDIR /app

# Copy application files
COPY skripsi/custom_pyktok /app/custom_pyktok
COPY skripsi/medsos_lrcn/src/worker.py /app
COPY skripsi/medsos_lrcn/src/all_config.py /app
COPY skripsi/medsos_lrcn/src/loader_data.py /app
COPY skripsi/medsos_lrcn/src/models_bidir.py /app
COPY skripsi/medsos_lrcn/src/testcv.py /app

# Playwright setup
RUN firefox --headless & (sleep 5 && kill $!) || true

EXPOSE 54000

CMD ["python3","worker.py"]

#docker run --gpus all --network backend-network -v /home/arifadh/




