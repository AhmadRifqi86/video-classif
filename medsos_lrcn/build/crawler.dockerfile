FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    firefox-esr \
    #headless rendering
    xvfb \
    dbus-x11 \
    # Playwright dependencies
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz \
    && tar -xvzf geckodriver-v0.30.0-linux64.tar.gz \
    && mv geckodriver /usr/local/bin/ \
    && rm geckodriver-v0.30.0-linux64.tar.gz

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY skripsi/medsos_lrcn/src/crawler.py /app
COPY skripsi/custom_pyktok /app/custom_pyktok

RUN pip install requests==2.32.3 beautifulsoup4==4.12.3 browser_cookie3==0.19.1
RUN pip install numpy==1.26.4 pandas==2.2.2 TikTokApi==6.5.2 playwright==1.37.0

# Install Playwright browsers
RUN playwright install-deps
RUN playwright install firefox

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_STAGE=prod

RUN firefox --headless & (sleep 5 && kill $!) || true

# Default command (replace with your specific crawler script)
CMD ["python", "crawler.py"]




#docker build -f skripsi/medsos_lrcn/build/crawler.dockerfile -t crawler .
#docker run --rm --network backend-network -v /home/arifadh/Downloads/tiktok_videos:/app/videos crawler python3 crawler.py
