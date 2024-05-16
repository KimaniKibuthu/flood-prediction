
# Flood Prediction API

## Overview

The Flood Prediction API is designed to predict the likelihood of floods based on weather data, as well as provide estimates for rainfall and river discharge. The API fetches weather data from the Open-Meteo API and uses a pre-trained machine learning model to generate predictions. This API can be integrated into a Unity environment for visualization purposes.

## Features

- Predict flood probability based on current weather data.
- Provide estimated rainfall and daily river discharge.

## Requirements
- Python 3.8+
- Poetry

## Project Organization

```
flood-prediction/
├── LICENSE     
├── README.md                  
├── Makefile 
├── app.py  
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
    │
    ├── data                     # Data engineering scripts.
    │   ├── build_features.py    
    │   ├── cleaning.py          
    │   ├── ingestion.py         
    │   ├── labeling.py          
    │   ├── splitting.py         
    │   └── validation.py        
    │
    ├── models                   # ML model engineering (a folder for each model).
    │   └── model1      
    │       ├── dataloader.py    
    │       ├── hyperparameters_tuning.py 
    │       ├── model.py         
    │       ├── predict.py       
    │       ├── preprocessing.py 
    │       └── train.py         
    │
    └── visualization        # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py        
        └── exploration.py       
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
    poetry install
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

### POST /flood_prediction

**Description**: Predict flood probability and provide rainfall and river discharge estimates.

**Request Body:**

- location (string): The name of the location to predict floods for.

**Response:**

- flood_probability (float): The predicted probability of a flood.
- estimated_rainfall (float): The estimated rainfall in millimeters.
- daily_river_discharge (float): The estimated daily river discharge.

**Example:**

```json
{
  "location": "Barisal" # Currently only supporting the 33 stations in Bangladesh
}
```	

**Response Example:**

```json

{
  "flood_probability": 0.75,
  "estimated_rainfall": 10.5,
  "daily_river_discharge": 120.0
}
```	

## Contributions

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
