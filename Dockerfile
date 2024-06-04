# Use a smaller base image
FROM python:3.11.5-slim AS build

# Set the working directory
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgomp1 \
&& rm -rf /var/lib/apt/lists/*

# Copy only the necessary files for installing dependencies
COPY poetry.lock pyproject.toml ./

# Install Poetry with specific version
RUN pip install poetry

# Install the project dependencies
RUN poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry install --no-interaction --no-ansi 

# Copy the FastAPI application code and additional files to the working directory
COPY flood_prediction_api.py .
COPY data/ data/
COPY configs/ configs/
COPY models/ models/
COPY src/ src/

# Use a minimal runtime image
FROM python:3.11.5-slim

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgomp1 \
&& rm -rf /var/lib/apt/lists/*
# Set the working directory
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/flood_prediction_api.py /app/
COPY --from=build /app/data /app/data
COPY --from=build /app/configs /app/configs
COPY --from=build /app/models /app/models
COPY --from=build /app/src /app/src

# Set the environment variable for Poetry
ENV POETRY_HTTP_TIMEOUT=3600
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port for FastAPI
EXPOSE 8000

# Set the entrypoint to run FastAPI
CMD ["uvicorn", "flood_prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]