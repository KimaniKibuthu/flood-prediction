# Use a specific version of the Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files for installing dependencies
COPY poetry.lock pyproject.toml ./

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
EXPOSE 8000

# Set the entrypoint to run FastAPI
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
