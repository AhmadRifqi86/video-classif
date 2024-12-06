# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables
#ENV APP_STAGE=prod
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container, does it need to copy all code file?
COPY skripsi/medsos_lrcn/src/backend.py /app
COPY skripsi/medsos_lrcn/src/all_config.py /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies #if batch processing
RUN pip install --upgrade pip
RUN pip install flask==3.1.0 pymongo==4.10.1
# Real-time dependency
RUN pip install pyzmq==26.2.0

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
#docker network create backend-network
#docker run -d --name mongodb --network backend-network -p 27017:27017 mongo:5.0 [if first time]
#docker run -d --name backend --network backend-network -p 5000:5000 backend [if first time]

#test endpoint:
#curl -X POST -H "Content-Type: application/json" -d '{"url": "http://example.com/video.mp4", "labels": "Safe"}' http://localhost:5000/classify
#curl -X GET "http://localhost:5000/video_labels?url=http://example.com/video.mp4"

#important command:
# docker ps -a ;list container and state
# docker image list ; list of image
# docker rmi  [image-id] ; delete an image
# docker start [container-name] ; start a container
# docker stop [container-name] ; stop a container
# docker rm [container-name] ; delete a container


