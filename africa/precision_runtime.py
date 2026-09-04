"""Africa runtime bridge to the precision-v5 score engine."""
from __future__ import annotations

from . import runtime_hardening as _rh
from precision_sim_engine import simulate_precision

# Keep Africa's hardened registry/model/mapping layer, but replace only the
# stochastic score layer with the deterministic precision engine.
_rh._ORIGINAL_SIMULATE_SCORES = simulate_precision
_ORIGINAL_BUILD = _rh._build_feature_row


def simulate_match(*args, **kwargs):
    match_date = kwargs.get("match_date")
    original_build = _rh._build_feature_row

    def fixture_clock_build(home, away, hist, now=None):
        # Prevent runner-date leakage for future/past fixtures.  The Africa
        # model state is always computed strictly before the fixture kickoff.
        clock = now
        if match_date:
            try:
                import pandas as pd
                parsed = pd.to_datetime(match_date, errors="coerce")
                if pd.notna(parsed):
                    clock = parsed.to_pydatetime().replace(tzinfo=None)
            except Exception:
                pass
        return original_build(home, away, hist, now=clock)

    _rh._build_feature_row = fixture_clock_build
    try:
        return _rh.simulate_match(*args, **kwargs)
    finally:
        _rh._build_feature_row = original_build
