"""Tests for CLI broker selection based on KAIRON_BYBIT_BROKER."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kairon.cli.trade import _run_trading_loop
from kairon.config import KaironSettings


def _run_once(settings: KaironSettings, *, dry_run: bool, testnet: bool) -> None:
    """Invoke the trading loop with all collaborators mocked.

    Exceptions are swallowed because the test only asserts which broker class was
    instantiated.
    """
    with (
        patch("kairon.live.feed.CcxtCandleFeed", return_value=MagicMock()),
        patch("kairon.live.orchestrator.TradingLoop", return_value=MagicMock()),
        patch("kairon.live.guardian.Guardian", return_value=MagicMock()),
        patch("kairon.live.reconciler.Reconciler", return_value=MagicMock()),
        patch("kairon.live.store.LiveStore", return_value=MagicMock()),
        patch("asyncio.run", return_value={"ok": True, "balances": []}),
        contextlib.suppress(Exception),
    ):
        _run_trading_loop(
            settings,
            dry_run=dry_run,
            testnet=testnet,
            live=False,
            db_path=Path("data/runs.db"),
        )


class TestBrokerSelection:
    """Verify the CLI instantiates the correct broker class."""

    @pytest.mark.asyncio
    async def test_default_selects_bybit_broker(self) -> None:
        settings = KaironSettings(
            bybit_api_key="key",
            bybit_api_secret="secret",  # noqa: S106
            bybit_broker="pybit",
            live_symbols=("BTC-USDT-PERP",),
        )

        with patch("kairon.live.broker.BybitBroker") as mock_broker:
            _run_once(settings, dry_run=False, testnet=True)
        assert mock_broker.called

    @pytest.mark.asyncio
    async def test_raw_selects_bybit_raw_broker(self) -> None:
        settings = KaironSettings(
            bybit_api_key="key",
            bybit_api_secret="secret",  # noqa: S106
            bybit_broker="raw",
            live_symbols=("BTC-USDT-PERP",),
        )

        with patch("kairon.live.broker.BybitRawBroker") as mock_broker:
            _run_once(settings, dry_run=False, testnet=True)
        assert mock_broker.called

    @pytest.mark.asyncio
    async def test_dry_run_uses_paper_broker(self) -> None:
        settings = KaironSettings(
            bybit_broker="raw",
            live_symbols=("BTC-USDT-PERP",),
        )

        with (
            patch("kairon.live.broker.PaperBroker") as mock_paper,
            patch("kairon.live.broker.BybitRawBroker") as mock_raw,
        ):
            _run_once(settings, dry_run=True, testnet=False)
            assert mock_paper.called
            assert not mock_raw.called


class TestFeedConfig:
    """Verify the candle feed config receives the correct testnet flag."""

    @pytest.mark.asyncio
    async def test_testnet_flag_matches_cli(self) -> None:
        settings = KaironSettings(
            bybit_api_key="key",
            bybit_api_secret="secret",  # noqa: S106
            bybit_broker="pybit",
            live_symbols=("BTC-USDT-PERP",),
        )

        with (
            patch("kairon.live.broker.BybitBroker", return_value=MagicMock()),
            patch("kairon.live.feed.CcxtCandleFeedConfig") as mock_config,
            patch("kairon.live.feed.CcxtCandleFeed", return_value=MagicMock()),
            patch("kairon.live.orchestrator.TradingLoop", return_value=MagicMock()),
            patch("kairon.live.guardian.Guardian", return_value=MagicMock()),
            patch("kairon.live.reconciler.Reconciler", return_value=MagicMock()),
            patch("kairon.live.store.LiveStore", return_value=MagicMock()),
            patch("asyncio.run", return_value={"ok": True, "balances": []}),
            contextlib.suppress(Exception),
        ):
            _run_trading_loop(
                settings,
                dry_run=False,
                testnet=True,
                live=False,
                db_path=Path("data/runs.db"),
            )

        assert mock_config.called
        assert mock_config.call_args.kwargs["testnet"] is True

    @pytest.mark.asyncio
    async def test_live_mode_uses_testnet_false(self) -> None:
        settings = KaironSettings(
            bybit_api_key="key",
            bybit_api_secret="secret",  # noqa: S106
            bybit_broker="pybit",
            live_symbols=("BTC-USDT-PERP",),
        )

        with (
            patch("kairon.live.broker.BybitBroker", return_value=MagicMock()),
            patch("kairon.live.feed.CcxtCandleFeedConfig") as mock_config,
            patch("kairon.live.feed.CcxtCandleFeed", return_value=MagicMock()),
            patch("kairon.live.orchestrator.TradingLoop", return_value=MagicMock()),
            patch("kairon.live.guardian.Guardian", return_value=MagicMock()),
            patch("kairon.live.reconciler.Reconciler", return_value=MagicMock()),
            patch("kairon.live.store.LiveStore", return_value=MagicMock()),
            patch("asyncio.run", return_value={"ok": True, "balances": []}),
            contextlib.suppress(Exception),
        ):
            _run_trading_loop(
                settings,
                dry_run=False,
                testnet=False,
                live=True,
                db_path=Path("data/runs.db"),
            )

        assert mock_config.called
        assert mock_config.call_args.kwargs["testnet"] is False
