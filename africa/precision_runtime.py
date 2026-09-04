"""Africa runtime bridge to the shared precision-v5 score engine."""
from __future__ import annotations

from . import runtime_hardening as _rh
from precision_sim_engine import simulate_precision

# runtime_hardening already owns the Africa-specific fixture/team registry,
# mapping, chronological feature contract, fallback registries and ensemble.
# Replace only its stochastic score layer with the shared deterministic engine.
_rh._ORIGINAL_SIMULATE_SCORES = simulate_precision


def simulate_match(*args, **kwargs):
    return _rh.simulate_match(*args, **kwargs)
