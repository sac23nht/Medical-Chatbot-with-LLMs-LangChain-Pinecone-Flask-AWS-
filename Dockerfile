FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Install system dependencies (optional, for packages that need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first (better Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Now copy the rest of the app
COPY . .

# Run the app
CMD ["python3", "app.py"]
