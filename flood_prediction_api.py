from src.utils import logger, load_config
from src.full_pipeline import main
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import RedirectResponse
import openmeteo_requests
from autogluon.tabular import TabularDataset, TabularPredictor
import requests_cache
import pandas as pd
from retry_requests import retry
import uvicorn
import asyncio
import pickle
import json
import os
import numpy as np
from slowapi import _rate_limit_exceeded_handler, Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
import redis
from functools import lru_cache
from requests import Session, Response
from typing import Any, Dict, List, Tuple, Union


class Settings(BaseSettings):
    redis_host: str
    redis_port: int
    redis_password: str
    open_meteo_url: str
    flood_api_url: str

    class Config:
        env_file = ".env"


settings = Settings()

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    username="default",
    password=settings.redis_password,
)

redis_client.flushdb()


class FloodPredictionRequest(BaseModel):
    location: str


class FloodPredictionResponse(BaseModel):
    current_flood_probability: float
    forecast_flood_probability: float
    estimated_rainfall: float
    daily_river_discharge: float


@lru_cache(maxsize=1024)
def cache_response(location: str, response: str) -> str:
    """
    Cache the response for a given location.

    Args:
        location (str): The location for which the response is being cached.
        response (str): The response to be cached.

    Returns:
        str: The cached response.
    """
    redis_client.set(location, response, ex=3600)
    return response


def load_model(config: Dict[str, Any] = load_config()) -> TabularPredictor:
    """
    Load the pre-trained model.

    Args:
        config (Dict[str, Any], optional): The configuration dictionary. Defaults to load_config().

    Returns:
        TabularPredictor: The loaded pre-trained model.
    """
    model_path = os.path.normpath(config["modelling"]["models_directory"])
    return TabularPredictor.load(model_path)


def load_reference_data(config: Dict[str, Any] = load_config()) -> pd.DataFrame:
    """
    Load the reference data.

    Args:
        config (Dict[str, Any], optional): The configuration dictionary. Defaults to load_config().

    Returns:
        pd.DataFrame: The reference data as a pandas DataFrame.
    """
    reference_data_path = os.path.normpath(config["data"]["reference_data_path"])
    return pd.read_csv(reference_data_path)


def validate_location(location: str, reference_data: pd.DataFrame) -> None:
    """
    Validate the location against the reference data.

    Args:
        location (str): The location to be validated.
        reference_data (pd.DataFrame): The reference data.

    Raises:
        HTTPException: If the location is not found in the reference data.
    """
    if location not in reference_data["Station_Names"].values:
        raise HTTPException(
            status_code=400,
            detail=f"Location '{location}' not found in reference data.",
        )


async def fetch_weather_data(location: str, retry_session: Session) -> Response:
    """
    Fetch current weather data for the given location.

    Args:
        location (str): The location for which to fetch weather data.
        retry_session (requests.Session): The session to use for making requests.

    Returns:
        openmeteo_requests.Response: The response containing the current weather data.
    """
    openmeteo = openmeteo_requests.Client(session=retry_session)
    reference_data = load_reference_data()
    supp_data = reference_data.loc[reference_data["Station_Names"] == location]
    longitude = supp_data.LONGITUDE_y.values[0]
    latitude = supp_data.LATITUDE_y.values[0]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "relative_humidity_2m", "rain", "cloud_cover"],
        "timezone": "auto",
    }
    return openmeteo.weather_api(settings.open_meteo_url, params=params)


async def fetch_forecast_weather_data(
    location: str, retry_session: Session
) -> Response:
    """
    Fetch forecast weather data for the given location.

    Args:
        location (str): The location for which to fetch forecast weather data.
        retry_session (requests.Session): The session to use for making requests.

    Returns:
        openmeteo_requests.Response: The response containing the forecast weather data.
    """
    openmeteo = openmeteo_requests.Client(session=retry_session)
    reference_data = load_reference_data()
    supp_data = reference_data.loc[reference_data["Station_Names"] == location]
    longitude = supp_data.LONGITUDE_y.values[0]
    latitude = supp_data.LATITUDE_y.values[0]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "cloud_cover"],
        "timezone": "auto",
        "forecast_days": 14,
    }
    return openmeteo.weather_api(settings.open_meteo_url, params=params)


async def fetch_river_discharge_data(location: str, retry_session: Session) -> Response:
    """
    Fetch river discharge data for the given location.

    Args:
        location (str): The location for which to fetch river discharge data.
        retry_session (requests.Session): The session to use for making requests.

    Returns:
        openmeteo_requests.Response: The response containing the river discharge data.
    """
    openmeteo = openmeteo_requests.Client(session=retry_session)
    reference_data = load_reference_data()
    supp_data = reference_data.loc[reference_data["Station_Names"] == location]
    longitude = supp_data.LONGITUDE_y.values[0]
    latitude = supp_data.LATITUDE_y.values[0]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "river_discharge",
        "forecast_days": 1,
    }
    return openmeteo.weather_api(settings.flood_api_url, params=params)


async def fetch_all_weather_data(
    location: str, retry_session: Session
) -> Tuple[Response, Response, Response]:
    """
    Fetch all weather data (current, forecast, and river discharge) for the given location.

    Args:
        location (str): The location for which to fetch weather data.
        retry_session (requests.Session): The session to use for making requests.

    Returns:
        Tuple[openmeteo_requests.Response, openmeteo_requests.Response, openmeteo_requests.Response]: A tuple containing the responses for current weather data, forecast weather data, and river discharge data, respectively.
    """
    tasks = [
        fetch_weather_data(location, retry_session),
        fetch_forecast_weather_data(location, retry_session),
        fetch_river_discharge_data(location, retry_session),
    ]
    return await asyncio.gather(*tasks)


def process_current_weather_data(
    response: Response, location: str
) -> Tuple[float, float, float, float, float, float, float, float, int, float, float]:
    """
    Process the current weather data response for the given location.

    Args:
        response (openmeteo_requests.Response): The response containing the current weather data.
        location (str): The location for which the weather data is being processed.

    Returns:
        Tuple[float, float, float, float, float, float, float, float, int, float, float]: A tuple containing the following values:
            - current_temperature_2m (float): Current temperature at 2m above ground level.
            - current_relative_humidity_2m (float): Current relative humidity at 2m above ground level.
            - current_rain (float): Current rainfall.
            - current_cloud_cover (float): Current cloud cover.
            - latitude (float): Latitude of the location.
            - longitude (float): Longitude of the location.
            - alt (float): Altitude of the location.
            - dist_to_water (float): Distance to the nearest water body.
            - station_number (int): Station number of the location.
            - rain_latitude (float): Rainfall multiplied by latitude.
            - rain_longitude (float): Rainfall multiplied by longitude.
    """
    reference_data = load_reference_data()
    supp_data = reference_data.loc[reference_data["Station_Names"] == location]
    latitude = supp_data.LATITUDE_y.values[0]
    longitude = supp_data.LONGITUDE_y.values[0]
    alt = supp_data.ALT.values[0]
    dist_to_water = supp_data.dist_to_water.values[0]
    station_number = supp_data.Station_Number.values[0]

    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_relative_humidity_2m = current.Variables(1).Value()
    current_rain = current.Variables(2).Value()
    current_cloud_cover = current.Variables(3).Value()
    rain_latitude = current_rain * latitude
    rain_longitude = current_rain * longitude

    return (
        current_temperature_2m,
        current_relative_humidity_2m,
        current_rain,
        current_cloud_cover,
        latitude,
        longitude,
        alt,
        dist_to_water,
        station_number,
        rain_latitude,
        rain_longitude,
    )


def process_forecast_weather_data(
    response: Response, location: str
) -> Tuple[float, float, float, float, int, float, float, float, float]:
    """
    Process the forecast weather data response for the given location.

    Args:
        response (openmeteo_requests.Response): The response containing the forecast weather data.
        location (str): The location for which the weather data is being processed.

    Returns:
        Tuple[float, float, float, float, int, float, float, float, float]: A tuple containing the following values:
            - min_temperature (float): Minimum temperature over the forecast period.
            - relative_humidity_2m (float): Average relative humidity at 2m above ground level over the forecast period.
            - rain (float): Total rainfall over the forecast period.
            - cloud_cover (float): Average cloud cover over the forecast period.
            - station_number (int): Station number of the location.
            - alt (float): Altitude of the location.
            - dist_to_water (float): Distance to the nearest water body.
            - rain_latitude (float): Total rainfall multiplied by latitude.
            - rain_longitude (float): Total rainfall multiplied by longitude.
    """
    reference_data = load_reference_data()
    supp_data = reference_data.loc[reference_data["Station_Names"] == location]
    latitude = supp_data.LATITUDE_y.values[0]
    longitude = supp_data.LONGITUDE_y.values[0]
    alt = supp_data.ALT.values[0]
    dist_to_water = supp_data.dist_to_water.values[0]
    station_number = supp_data.Station_Number.values[0]

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["rain"] = hourly_rain
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    resampled_data = hourly_dataframe.resample("14D", on="date").agg(
        {
            "temperature_2m": "mean",
            "relative_humidity_2m": "mean",
            "rain": "sum",
            "cloud_cover": "mean",
        }
    )

    min_temperature = resampled_data["temperature_2m"].values[0]
    relative_humidity_2m = resampled_data["relative_humidity_2m"].values[0]
    rain = resampled_data["rain"].values[0]
    cloud_cover = resampled_data["cloud_cover"].values[0]
    rain_latitude = rain * latitude
    rain_longitude = rain * longitude

    return (
        min_temperature,
        relative_humidity_2m,
        rain,
        cloud_cover,
        station_number,
        alt,
        dist_to_water,
        rain_latitude,
        rain_longitude,
    )


def process_river_discharge_data(response: Response) -> np.ndarray:
    """
    Process the river discharge data response.

    Args:
        response (openmeteo_requests.Response): The response containing the river discharge data.

    Returns:
        np.ndarray: A numpy array containing the daily river discharge data.
    """
    daily = response.Daily()
    daily_river_discharge = daily.Variables(0).ValuesAsNumpy()
    return daily_river_discharge


def make_prediction(
    model: TabularPredictor, feature_values: List[Union[float, int]]
) -> float:
    """
    Make a flood prediction using the given model and feature values.

    Args:
        model (TabularPredictor): The pre-trained model to use for prediction.
        feature_values (List[Union[float, int]]): A list of feature values for the prediction.

    Returns:
        float: The predicted flood probability.
    """
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
    data = {feature: [value] for feature, value in zip(feature_names, feature_values)}
    df = pd.DataFrame(data)
    predict_df = TabularDataset(df)
    return model.predict_proba(predict_df)[1][0]


def load_and_cache_model(config: Dict[str, Any] = load_config()) -> TabularPredictor:
    """
    Load and cache the pre-trained model.

    Args:
        config (Dict[str, Any], optional): The configuration dictionary. Defaults to load_config().

    Returns:
        TabularPredictor: The loaded and cached pre-trained model.
    """
    model_key = "flood_prediction_model"
    model_bytes = redis_client.get(model_key)
    if model_bytes:
        model = pickle.loads(model_bytes)
    else:
        model_path = os.path.normpath(config["modelling"]["models_directory"])
        model = TabularPredictor.load(model_path)
        model_bytes = pickle.dumps(model)
        redis_client.set(model_key, model_bytes)
    return model


app = FastAPI(
    title="Flood Prediction API",
    description="An API to predict flood probabilities using weather data.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.get("/", tags=["Root"])
async def read_root() -> RedirectResponse:
    """
    Redirect the root endpoint to the documentation.

    Returns:
        RedirectResponse: A redirect response to the documentation endpoint.
    """
    return RedirectResponse(url="/docs")


@app.post(
    "/v1/flood-prediction",
    response_model=FloodPredictionResponse,
    tags=["Flood Prediction"],
)
async def flood_prediction(request: FloodPredictionRequest) -> FloodPredictionResponse:
    """
    Predict flood probabilities based on the given location.

    Args:
        request (FloodPredictionRequest): The request containing the location.

    Returns:
        FloodPredictionResponse: The response containing the current and forecast flood probabilities, estimated rainfall, and daily river discharge.

    Raises:
        HTTPException: If the location is not found in the reference data or if an unexpected error occurs.
        RateLimitExceeded: If the rate limit is exceeded.
    """
    try:
        location = request.location
        reference_data = load_reference_data()
        validate_location(location, reference_data)

        cached_response = redis_client.get(location)
        if cached_response:
            response_dict = json.loads(cached_response)
            response = FloodPredictionResponse(**response_dict)
            return response

        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

        (
            current_weather,
            forecast_weather,
            river_discharge,
        ) = await fetch_all_weather_data(location, retry_session)

        current_weather_data = process_current_weather_data(
            current_weather[0], location
        )
        forecast_weather_data = process_forecast_weather_data(
            forecast_weather[0], location
        )
        daily_river_discharge = process_river_discharge_data(river_discharge[0])

        model = load_and_cache_model()

        current_flood_probability = make_prediction(model, current_weather_data[:-2])
        forecast_flood_probability = make_prediction(model, forecast_weather_data)

        response = FloodPredictionResponse(
            current_flood_probability=current_flood_probability,
            forecast_flood_probability=forecast_flood_probability,
            estimated_rainfall=current_weather_data[2],
            daily_river_discharge=daily_river_discharge,
        )
        logger.info("Prediction successful")
        cache_response(location, response.json())
        return response

    except RateLimitExceeded:
        raise HTTPException(
            status_code=429, detail="Too many requests. Please try again later."
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred."
        ) from e


@app.post("/v1/train-model", tags=["Model Training"])
async def train_model() -> Dict[str, str]:
    """
    Train the flood prediction model.

    Returns:
        Dict[str, str]: A dictionary containing a success message.

    Raises:
        HTTPException: If an unexpected error occurs during training.
    """
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
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
