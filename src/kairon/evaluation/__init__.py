"""Evaluation utilities (cost calibration, break-even tables, sensitivity).

This package is intentionally side-effect-free. It exists so that
downstream modules (backtest engine, paper simulator, evaluation
harness) can import a single place for impact-model calibration and
performance decomposition. Story W2.1 ships the Almgren-Chriss
``eta`` calibrator against (price, qty, adv, sigma) trade tuples.
"""
