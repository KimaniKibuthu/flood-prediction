from src.utils import logger, load_config
from src.full_pipeline import main
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse
import openmeteo_requests
from autogluon.tabular import TabularDataset, TabularPredictor
import requests_cache
import pandas as pd
from retry_requests import retry
import uvicorn
import asyncio


app = FastAPI(
    title="Flood Prediction API",
    description="An API to predict floods based on weather data and provide rainfall and river discharge estimates.",
    version="0.0.1",
)


# Enable caching for external requests
requests_cache.install_cache(".cache", expire_after=3600)


class FloodPredictionRequest(BaseModel):
    location: str


class FloodPredictionResponse(BaseModel):
    flood_probability: float
    estimated_rainfall: float
    daily_river_discharge: float


def load_model(config=load_config()):
    return TabularPredictor.load(config["modelling"]["models_directory"])


def load_reference_data(config=load_config()):
    return pd.read_csv(config["data"]["reference_data_path"])


@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    return RedirectResponse(url="/docs")


@app.post(
    "/v1/flood-prediction",
    response_model=FloodPredictionResponse,
    tags=["Flood Prediction"],
)
async def flood_prediction(request: FloodPredictionRequest):
    try:
        # Fetch weather data from Open-Meteo
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        reference_data = load_reference_data()
        location = request.location

        # Validate location
        if location not in reference_data["Station_Names"].values:
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' not found in reference data.",
            )

        supp_data = reference_data.loc[reference_data["Station_Names"] == location]
        longitude = supp_data.LONGITUDE_y.values[0]
        latitude = supp_data.LATITUDE_y.values[0]
        alt = supp_data.ALT.values[0]
        dist_to_water = supp_data.dist_to_water.values[0]
        station_number = supp_data.Station_Number.values[0]

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "rain",
                "cloud_cover",
            ],
            "timezone": "auto",
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Current values. The order of variables needs to be the same as requested.
        current = response.Current()
        current_temperature_2m = current.Variables(0).Value()
        current_relative_humidity_2m = current.Variables(1).Value()
        current_rain = current.Variables(2).Value()
        current_cloud_cover = current.Variables(3).Value()
        rain_latitude = current_rain * latitude
        rain_longitude = current_rain * longitude

        # Load the model
        model = load_model()

        # Make predictions using the loaded model
        feature_names = [
            "Min_Temp",
            "Rainfall",
            "Cloud_Coverage",
            "Station_Number",
            "ALT",
            "rain_latitude",
            "rain_longitude",
            "dist_to_water",
        ]
        feature_values = [
            current_temperature_2m,
            current_rain,
            current_cloud_cover,
            station_number,
            alt,
            rain_latitude,
            rain_longitude,
            dist_to_water,
        ]
        data = {
            feature: [value] for feature, value in zip(feature_names, feature_values)
        }
        df = pd.DataFrame(data)
        predict_df = TabularDataset(df)
        flood_probability = model.predict_proba(predict_df)[1][0]

        # Get river discharge
        flood_url = "https://flood-api.open-meteo.com/v1/flood"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "river_discharge",
            "forecast_days": 1,
        }
        responses = openmeteo.weather_api(flood_url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_river_discharge = daily.Variables(0).ValuesAsNumpy()

        # Return the response
        return FloodPredictionResponse(
            flood_probability=flood_probability,
            estimated_rainfall=current_rain,
            daily_river_discharge=daily_river_discharge,
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred."
        ) from e


@app.post("/v1/train-model", tags=["Model Training"])
async def train_model():
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, main)
        return {"message": "Model Trained Successfully"}
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during training."
        ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
