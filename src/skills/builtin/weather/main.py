"""Weather skill — query weather forecast via wttr.in (free, no API key)."""

from __future__ import annotations

import httpx


def run(city: str, days: int = 2) -> dict:
    """查询天气预报。

    Args:
        city: 城市名称（支持中英文）
        days: 预报天数（1-3）

    Returns:
        包含当前天气和预报的字典
    """
    url = f"https://wttr.in/{city}"
    params = {"format": "j1"}

    resp = httpx.get(url, params=params, timeout=15, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()

    # Current conditions
    current = data.get("current_condition", [{}])[0]
    current_info = {
        "temperature": current.get("temp_C", "?"),
        "feels_like": current.get("FeelsLikeC", "?"),
        "humidity": current.get("humidity", "?"),
        "wind_speed": current.get("windspeedKmph", "?"),
        "wind_dir": current.get("winddir16Point", "?"),
        "description": current.get("lang_zh", [{}])[0].get("value", current.get("weatherDesc", [{}])[0].get("value", "?")),
        "visibility": current.get("visibility", "?"),
        "pressure": current.get("pressure", "?"),
        "uv_index": current.get("uvIndex", "?"),
    }

    # Forecast
    forecast_raw = data.get("weather", [])[:days]
    forecast = []
    for day in forecast_raw:
        forecast.append({
            "date": day.get("date", "?"),
            "high": day.get("maxtempC", "?"),
            "low": day.get("mintempC", "?"),
            "description": day.get("hourly", [{}])[4].get("lang_zh", [{}])[0].get("value", "?") if len(day.get("hourly", [])) > 4 else "?",
            "sunrise": day.get("astronomy", [{}])[0].get("sunrise", "?") if day.get("astronomy") else "?",
            "sunset": day.get("astronomy", [{}])[0].get("sunset", "?") if day.get("astronomy") else "?",
        })

    location = data.get("nearest_area", [{}])[0]
    location_name = location.get("areaName", [{}])[0].get("value", city)
    country = location.get("country", [{}])[0].get("value", "")

    return {
        "location": f"{location_name}, {country}" if country else location_name,
        "current": current_info,
        "forecast": forecast,
    }
