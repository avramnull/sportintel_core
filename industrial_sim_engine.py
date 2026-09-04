#!/usr/bin/env python3
"""Industrial compatibility layer for the precision-v5 simulation engine.

The public ``simulate_industrial`` API is retained so existing normal-fixture
pipeline code continues to work, while all published probabilities now come
from the deterministic precision engine.
"""
from __future__ import annotations

from precision_sim_engine import (
    ENGINE_VERSION,
    simulate_precision,
    simulate_industrial,
)

__all__ = ["ENGINE_VERSION", "simulate_precision", "simulate_industrial"]
