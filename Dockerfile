# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Ensure Python output is logged immediately and disable pip cache
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies needed for Selenium and Firefox
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        firefox-esr \
        wget \
        unzip \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install geckodriver (version 0.33.0)
ENV GECKODRIVER_VERSION=0.33.0
RUN wget -q "https://github.com/mozilla/geckodriver/releases/download/v${GECKODRIVER_VERSION}/geckodriver-v${GECKODRIVER_VERSION}-linux64.tar.gz" && \
    tar -xzf "geckodriver-v${GECKODRIVER_VERSION}-linux64.tar.gz" -C /usr/local/bin && \
    rm "geckodriver-v${GECKODRIVER_VERSION}-linux64.tar.gz"

# Set the working directory in the container
WORKDIR /app

# Add /app to PYTHONPATH so Python can find the src/ directory
ENV PYTHONPATH=/app

# Create a non-root user for running the app
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Add /home/appuser/.local/bin to PATH so that installed scripts are found
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Copy Python dependencies and install them
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Set the default command to run your pipeline with the production configuration
CMD ["python", "src/pipeline/pipeline_creation.py", "--config", "src/config/production_config.yaml"]
