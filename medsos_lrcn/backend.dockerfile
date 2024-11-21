# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container, does it need to copy all code file?
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 5000 for the Flask application
EXPOSE 5000

# Run the application
CMD ["python", "backend.py"]


#build
#docker build -f [dockerfile name]-t [image name] .

#run
#docker run -p 5000:5000 -d [image name]


#run mongo
#docker pull mongo
#docker run --name mongodb --network flask-mongo-network -d mongo

