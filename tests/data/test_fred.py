"""Tests for the FRED macro adapter — fully mocked, no real network call."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
import respx
from httpx import Response

from kairon.data.adapters import AdapterError
from kairon.data.adapters.fred import FRED_SCHEMA, FredAdapter
from kairon.data.symbols import index_


def _fred_response(observations: list[dict[str, str]]) -> dict[str, Any]:
    return {"observations": observations}


@respx.mock
def test_fetch_parses_observations() -> None:
    obs = [
        {"date": "2024-01-01", "value": "3.4"},
        {"date": "2024-02-01", "value": "3.5"},
        {"date": "2024-03-01", "value": "."},  # missing
        {"date": "not-a-date", "value": "3.6"},  # bad
    ]
    respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
        return_value=Response(200, json=_fred_response(obs))
    )
    adapter = FredAdapter(api_key="DUMMY")
    table = adapter.fetch(
        index_("CPIAUCSL"),
        "1d",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 4, 1, tzinfo=UTC),
    )
    assert table.schema == FRED_SCHEMA
    assert table.num_rows == 2  # only valid rows kept


@respx.mock
def test_fetch_raises_on_429() -> None:
    respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
        return_value=Response(429, text="rate limited")
    )
    adapter = FredAdapter(api_key="DUMMY")
    with pytest.raises(AdapterError, match="rate limited"):
        adapter.fetch(
            index_("CPIAUCSL"),
            "1d",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 4, 1, tzinfo=UTC),
        )


@respx.mock
def test_fetch_raises_on_500() -> None:
    respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
        return_value=Response(500, text="boom")
    )
    adapter = FredAdapter(api_key="DUMMY")
    with pytest.raises(AdapterError, match="HTTP 500"):
        adapter.fetch(
            index_("CPIAUCSL"),
            "1d",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 4, 1, tzinfo=UTC),
        )


@respx.mock
def test_fetch_raises_on_unexpected_payload() -> None:
    respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
        return_value=Response(200, json={"error": "weird"})
    )
    adapter = FredAdapter(api_key="DUMMY")
    with pytest.raises(AdapterError, match="unexpected response"):
        adapter.fetch(
            index_("CPIAUCSL"),
            "1d",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 4, 1, tzinfo=UTC),
        )


def test_ctor_requires_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True), pytest.raises(AdapterError, match="FRED_API_KEY"):
        FredAdapter()


def test_ctor_reads_api_key_from_env() -> None:
    with patch.dict("os.environ", {"FRED_API_KEY": "env-key"}):
        a = FredAdapter()
        assert a.api_key == "env-key"
