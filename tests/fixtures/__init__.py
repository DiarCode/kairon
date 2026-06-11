"""Shared test fixtures for the kairon test suite.

This package is intentionally side-effect-free. It exists so that
downstream test modules can do::

    from tests.fixtures.leakage import (
        real_history_fixture,
        assert_no_leakage,
        assert_timestamp_monotonic,
    )

without re-implementing the helpers.
"""
