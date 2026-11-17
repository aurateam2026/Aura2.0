# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import datetime as dt
import json

import pytest
import requests

from examples.online_serving.aura_v2_gateway.gateway import native_tools

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _run(**kwargs):
    return asyncio.run(native_tools.NativeGatewayTools().run("get_city_weather", json.dumps(kwargs)))


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def _mock_weather(monkeypatch, payload):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if "geocoding-api" in url:
            return _Response({"results": [{"name": "香港", "latitude": 22.28, "longitude": 114.17}]})
        if isinstance(payload, Exception):
            raise payload
        return _Response(payload)

    monkeypatch.setattr(native_tools.requests, "get", get)
    return calls


def test_weather_connection_failure_is_not_success(monkeypatch):
    _mock_weather(monkeypatch, requests.ConnectionError("Connection reset by peer"))
    result = _run(city="香港")
    assert result == {
        "status": "error",
        "output": "",
        "error": "Weather lookup failed: Connection reset by peer",
    }


@pytest.mark.parametrize("days", [0, 1])
def test_forecast_override_and_clear_sky_code(monkeypatch, days):
    date = str(dt.date.today() + dt.timedelta(days=days))
    endpoint = "https://weather.example/v1/forecast"
    monkeypatch.setenv("AURA_WEATHER_FORECAST_URL", endpoint)
    calls = _mock_weather(
        monkeypatch,
        {"daily": {"time": [date], "weather_code": [0], "temperature_2m_max": [25]}},
    )
    result = _run(city="香港", date=date)
    assert calls[-1] == endpoint
    assert result["status"] == "success"
    output = json.loads(result["output"])
    assert output["date"] == date
    assert output["weather_code"] == 0
    assert output["weather_text"] == "晴朗"


def test_current_weather_uses_override(monkeypatch):
    endpoint = "https://weather.example/v1/forecast"
    monkeypatch.setenv("AURA_WEATHER_FORECAST_URL", endpoint)
    calls = _mock_weather(monkeypatch, {"current": {"weather_code": 0, "temperature_2m": 23}})
    result = _run(city="香港")
    assert calls[-1] == endpoint
    assert result["status"] == "success"
    assert json.loads(result["output"])["temperature"] == 23


def test_past_weather_still_uses_archive(monkeypatch):
    date = str(dt.date.today() - dt.timedelta(days=10))
    monkeypatch.setenv("AURA_WEATHER_FORECAST_URL", "https://weather.example/v1/forecast")
    calls = _mock_weather(monkeypatch, {"daily": {"time": [date], "weather_code": [3]}})
    assert _run(city="香港", date=date)["status"] == "success"
    assert calls[-1] == "https://archive-api.open-meteo.com/v1/archive"


@pytest.mark.parametrize("daily", [{}, {"time": ["2000-01-01"], "weather_code": [0]}])
def test_missing_or_wrong_date_is_not_success(monkeypatch, daily):
    _mock_weather(monkeypatch, {"daily": daily})
    result = _run(city="香港", date=str(dt.date.today()))
    assert result["status"] == "error"
    assert "未找到" in result["error"]


def test_invalid_date_does_not_make_network_requests(monkeypatch):
    calls = _mock_weather(monkeypatch, {})
    assert _run(city="香港", date="tomorrow")["status"] == "error"
    assert calls == []
