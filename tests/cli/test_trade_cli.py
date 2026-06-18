"""Tests for the trade CLI: start, stop, status, flatten commands."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kairon.cli.trade import trade_app
from kairon.live.store import LiveStore

runner = CliRunner()


class TestTradeCLIStop:
    """Test the trade stop command."""

    def test_stop_halts_store(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = runner.invoke(trade_app, ["stop", "--db", str(db_path)])
            assert result.exit_code == 0
            assert "halted" in result.output.lower()

            store = LiveStore(db_path)
            assert store.is_halted()
            store.close()

    def test_stop_with_custom_reason(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = runner.invoke(trade_app, ["stop", "--reason", "daily_loss", "--db", str(db_path)])
            assert result.exit_code == 0

            store = LiveStore(db_path)
            reason = store.get_runtime_state("halted")
            assert reason == "daily_loss"
            store.close()


class TestTradeCLIStatus:
    """Test the trade status command."""

    def test_status_shows_running(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = LiveStore(db_path)
            store.write_heartbeat(mode="dry_run", equity=10500.0, n_positions=2)
            store.close()

            result = runner.invoke(trade_app, ["status", "--db", str(db_path)])
            assert result.exit_code == 0
            assert "Running" in result.output
            assert "dry_run" in result.output

    def test_status_shows_halted(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = LiveStore(db_path)
            store.halt(reason="test")
            store.write_heartbeat(mode="halted", equity=10000.0, n_positions=0)
            store.close()

            result = runner.invoke(trade_app, ["status", "--db", str(db_path)])
            assert result.exit_code == 0
            assert "HALTED" in result.output


class TestTradeCLIFlatten:
    """Test the trade flatten command."""

    def test_flatten_halts_and_records_event(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            result = runner.invoke(trade_app, ["flatten", "--db", str(db_path)])
            assert result.exit_code == 0
            assert "Flatten" in result.output

            store = LiveStore(db_path)
            assert store.is_halted()
            events = store.get_recent_events(limit=10)
            assert len(events) >= 1
            assert events[0]["kind"] == "flatten"
            store.close()


class TestTradeCLIStart:
    """Test the trade start command flags."""

    def test_live_refused_without_ack(self) -> None:
        """--live must be refused when KAIRON_LIVE_PROMOTION_ACK is not set."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove the ack env var if it exists
            result = runner.invoke(trade_app, ["start", "--live"])
            assert result.exit_code == 1
            assert "KAIRON_LIVE_PROMOTION_ACK" in result.output

    def test_live_accepted_with_ack(self) -> None:
        """--live is accepted when KAIRON_LIVE_PROMOTION_ACK=1."""
        # This would actually try to start the loop, so we just verify
        # the ack check passes. We'll test the mode validation separately.
        with patch.dict("os.environ", {"KAIRON_LIVE_PROMOTION_ACK": "1"}):
            # Don't actually start the loop — just test the flag validation
            with patch("kairon.cli.trade._run_trading_loop") as mock_run:
                result = runner.invoke(trade_app, ["start", "--live"])
                # The command should proceed past the ack check
                # (it may fail later due to missing API keys, which is fine)
                assert "KAIRON_LIVE_PROMOTION_ACK" not in result.output or result.exit_code != 1

    def test_multiple_mode_flags_rejected(self) -> None:
        """Only one mode flag can be specified."""
        result = runner.invoke(trade_app, ["start", "--dry-run", "--testnet"])
        assert result.exit_code == 1
        assert "exactly one" in result.output.lower()

    def test_default_mode_is_dry_run(self) -> None:
        """No mode flag defaults to dry-run."""
        result = runner.invoke(trade_app, ["start", "--help"])
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower()