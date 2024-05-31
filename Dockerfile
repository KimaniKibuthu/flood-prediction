# Use a specific version of the Python image
FROM python:3.11.5-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Copy only the necessary files for installing dependencies
COPY poetry.lock pyproject.toml ./

ENV POETRY_HTTP_TIMEOUT=3600 

# Install Poetry with specific version
RUN pip install poetry

# Install the project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root

# Copy the FastAPI application code and additional files to the working directory
COPY app.py .
COPY data/ data/ 
COPY configs/ configs/
COPY models/ models/
COPY src/ src/

# Expose the port for FastAPI
EXPOSE 8080

# Set the entrypoint to run FastAPI
CMD ["poetry", "run", "uvicorn", "flood_prediction_api:app", "--host", "0.0.0.0", "--port", "8080"]