
# Flood Prediction API

## Overview

The Flood Prediction API is designed to predict the likelihood of floods based on weather data, as well as provide estimates for rainfall and river discharge. The API fetches weather data from the Open-Meteo API and uses a pre-trained machine learning model to generate predictions. This API can be integrated into a Unity environment for visualization purposes.

## Features

- Predict flood probability based on current weather data.
- Provide estimated rainfall and daily river discharge.

## Requirements
- Python 3.11
- Poetry

## Project Organization

```
flood-prediction/
├── LICENSE     
├── README.md                  
├── Makefile 
├── app.py  
├── Dockerfile
├── pyproject.toml                                      
├── configs                      
│   └── configs.yaml               # Configurations for the project   
│
├── data                         
│   ├── testing_data                
│   ├── training_data                  
│   ├── processed                
│   ├── raw                      
│   └── reference_data
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
│
├── references                   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                      # Generated analysis 
│   └── figures                  # Generated graphics and figures to be used in reporting.
│
└── src                          # Source code for use in this project.
    ├── __init__.py              # Makes src a Python module.            
    ├── build_features.py    
    ├── cleaning.py          
    ├── ingestion.py         
    ├── monitoring.py          
    ├── splitting.py         
    ├── full_pipeline.py           
    ├── train.py         

```
**DISCLAIMER**: As the project is still underway, all the work done is currently in the `notebooks` directory, the API in the `app.py` file and the model in the `models` directory. 
## Installation

1. Clone the repository:

2. Install Poetry:

    ```bash	
    pip install poetry
    ```	

3. Install dependencies:

    ```bash
    poetry install --no-root
    ```

4. Activate the virtual environment:

    ```bash
    poetry shell
    ```	

5. Start the API server:

    ```bash
    poetry run uvicorn app:app --host 0.0.0.0 --port 8000
    ```	

6. Access the API:

    Open your web browser and navigate to http://localhost:8000/docs to view and interact with the API.



## API Endpoints

### GET /

**Description**: Redirects to the API documentation.

**Response:**
- Redirects to `/docs`

### GET /v1/locations

**Description**: Get the list of locations covered by the model's scope.

**Response:**
- `locations` (list): A list of location names covered by the model.

**Example Response:**

```json
{
  "locations": [
    "Barisal",
    "Bhola",
    "Bogra",
    "Chandpur",
    "Chittagong (City-Ambagan)",
    "Chittagong (IAP-Patenga)",
    "Comilla",
    "Cox's Bazar",
    "Dhaka",
    "Dinajpur",
    "Faridpur",
    "Feni",
    "Hatiya",
    "Ishurdi",
    "Jessore",
    "Khepupara",
    "Khulna",
    "Kutubdia",
    "Madaripur",
    "Maijdee Court",
    "Mongla",
    "Mymensingh",
    "Patuakhali",
    "Rajshahi",
    "Rangamati",
    "Rangpur",
    "Sandwip",
    "Satkhira",
    "Sitakunda",
    "Srimangal",
    "Sylhet",
    "Tangail",
    "Teknaf"
  ]
}
```

### POST /v1/flood-prediction

**Description**: Predict flood probabilities based on the given location and provide rainfall and river discharge estimates.

**Request Body:**
- `location` (string): The name of the location to predict floods for. The location must be one of the locations listed in the `/v1/locations` endpoint.

**Response:**
- `current_flood_probability` (float): The predicted probability of a flood based on current weather data.
- `forecast_flood_probability` (float): The predicted probability of a flood based on forecast weather data.
- `estimated_rainfall` (float): The estimated rainfall in millimeters based on current weather data.
- `daily_river_discharge` (float): The estimated daily river discharge based on forecast weather data.

**Example Request:**

```json
{
  "location": "Barisal"
}
```

**Example Response:**

```json
{
  "current_flood_probability": 0.75,
  "forecast_flood_probability": 0.82,
  "estimated_rainfall": 10.5,
  "daily_river_discharge": 120.0
}
```


### POST /v1/train-model

**Description**: Train the flood prediction model.

**Response:**
- `message` (string): A success message indicating that the model was trained successfully.

**Example Response:**

```json
{
  "message": "Model Trained Successfully"
}
```



## Contributions

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
