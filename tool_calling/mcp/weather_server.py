from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Weather MCP Server")


@mcp.tool()
def get_weather(latitude: float, longitude: float) -> dict:
    """
    Get current weather for a location using latitude and longitude.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    current = data.get("current", {})

    return {
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "unit": {
            "temperature": "°C",
            "humidity": "%",
            "wind_speed": "km/h"
        }
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")