import requests
import os
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys - set via environment variables
ORS_API_KEY = os.getenv("ORS_API_KEY", "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImEyMjE5MGMwMTE0NzQyNTJiMzdlOGM1OWI2MzdlNWVlIiwiaCI6Im11cm11cjY0In0=")
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "46cb7beffaef6769000d273109b6e6ac")#

# Set to True for prototype/testing without real API keys
#DUMMY_MODE = os.getenv("DUMMY_MODE", "False").lower() == "true"
DUMMY_MODE = False


def geocode_location(place_name: str) -> Tuple[float, float]:
    """
    Convert a place name to coordinates using OpenRouteService geocoding API.
    
    Args:
        place_name (str): Name of the location (e.g., "Delhi")
        
    Returns:
        tuple: (lon, lat)
    """
    if DUMMY_MODE:
        logger.info("DUMMY MODE: Returning mock coordinates for %s", place_name)
        return 77.2090, 28.6139  # Delhi

    if ORS_API_KEY == "YOUR_ORS_API_KEY_HERE":
        raise ValueError("ORS API key not configured. Set ORS_API_KEY or enable DUMMY_MODE=true")

    url = "https://api.openrouteservice.org/geocode/search"
    params = {"api_key": ORS_API_KEY, "text": place_name, "size": 1}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("features"):
            raise Exception(f"Could not geocode place: {place_name}")
        coords = data["features"][0]["geometry"]["coordinates"]
        return coords[0], coords[1]  # lon, lat
    except Exception as e:
        logger.error(f"Error geocoding {place_name}: {e}")
        raise


def get_route_data(origin: Tuple[float, float], destination: Tuple[float, float]) -> Dict[str, Optional[int]]:
    if DUMMY_MODE:
        logger.info("DUMMY MODE: Returning mock route data")
        return {"distance_meters": 10000, "duration_sec": 900}

    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {"coordinates": [list(origin), list(destination)]}

    try:
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"ORS response: {data}")

        # ✅ handle new schema
        if "routes" in data and data["routes"]:
            route = data["routes"][0]
            return {
                "distance_meters": route["summary"]["distance"],
                "duration_sec": route["summary"]["duration"]
            }

        # (fallback for old schema, if ever returned)
        if "features" in data and data["features"]:
            route = data["features"][0]["properties"]["segments"][0]
            return {
                "distance_meters": route["distance"],
                "duration_sec": route["duration"]
            }

        raise Exception(f"ORS response has no route info: {data}")
    except Exception as e:
        logger.error(f"Error fetching ORS route: {e}")
        raise



def get_weather_data(lat: float, lon: float) -> Dict[str, float]:
    """Fetch current weather from OpenWeather API"""
    if DUMMY_MODE:
        logger.info("DUMMY MODE: Returning mock weather data")
        return {"temperature_c": 25.0, "precip_mm": 0.0, "wind_speed_mps": 3.5}

    if WEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY_HERE":
        raise ValueError("OpenWeather API key not configured. Set OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        temp = float(data["main"]["temp"])
        wind_speed = float(data["wind"]["speed"])
        precip_mm = 0.0
        if "rain" in data and "1h" in data["rain"]:
            precip_mm += float(data["rain"]["1h"])
        if "snow" in data and "1h" in data["snow"]:
            precip_mm += float(data["snow"]["1h"])
        return {"temperature_c": temp, "precip_mm": precip_mm, "wind_speed_mps": wind_speed}
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        raise


def get_complete_route_weather_data(origin_name: str, destination_name: str) -> Dict:
    """
    Convenience function: geocode place names, fetch route + weather data.
    
    Args:
        origin_name (str): Starting place name
        destination_name (str): Destination place name
    
    Returns:
        Dict containing route info, weather at midpoint, and midpoint coords
    """
    try:
        # Geocode
        origin_coords = geocode_location(origin_name)
        destination_coords = geocode_location(destination_name)

        # Route
        route_data = get_route_data(origin_coords, destination_coords)

        # Midpoint
        mid_lon = (origin_coords[0] + destination_coords[0]) / 2
        mid_lat = (origin_coords[1] + destination_coords[1]) / 2

        # Weather at midpoint
        weather_data = get_weather_data(mid_lat, mid_lon)

        return {"route": route_data, "weather": weather_data, "midpoint": {"lat": mid_lat, "lon": mid_lon}}
    except Exception as e:
        logger.error(f"Error getting complete route weather data: {e}")
        raise
