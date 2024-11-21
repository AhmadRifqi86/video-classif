# Use a base image with Python and Playwright support
FROM mcr.microsoft.com/playwright/python:v1.39.0-focal

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libnss3 \
    libatk1.0-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    libx11-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the Python script and all required files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ensure Playwright dependencies are installed
RUN playwright install --with-deps firefox

# Set the entry point for the container
ENTRYPOINT ["python3", "crawler.py"]

#build
#docker build -f [Dockerfile name] -t [Image name] .

#run
#docker run -v ~/.mozilla:/root/.mozilla -it --rm [Image name]

