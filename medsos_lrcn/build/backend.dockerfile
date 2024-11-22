# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container, does it need to copy all code file?
COPY skripsi/medsos_lrcn/src/backend.py /app
COPY skripsi/medsos_lrcn/src/all_config.py /app
COPY skripsi/medsos_lrcn/build/backend_req.txt /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r ./backend_req.txt

# Expose port 5000 for the Flask application
EXPOSE 5000

# Run the application
CMD ["python", "backend.py"]


#build
#docker build -f [dockerfile name]-t [image name] .
#example
#cd ~/Desktop/Skripsi-Magang-Proyek/
#docker build -f skripsi/medsos_lrcn/build/backend.dockerfile -t backend .


#run
#docker run -p 5000:5000 -d backend


#run mongo
#docker pull mongo
#docker run --name backend --network flask-mongo-network -p 5000:5000 -d backend



#important command:
# docker ps -a ;list container and state
# docker image list ; list of image
# docker rmi  [image-id] ; delete an image
# docker start [container-name] ; start a container
# docker stop [container-name] ; stop a container
# docker rm [container-name] ; delete a container


