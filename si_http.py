#!/usr/bin/env python3
"""
Industrial-grade HTTP download helper for sportintel_core.

Features:
  - Session with connection pooling + keep-alive
  - Exponential backoff retries (transient network / 5xx / timeouts)
  - Configurable timeout (connect + read)
  - Strong User-Agent
  - Content validation (CSV / min size / HTML rejection)
  - Atomic write (tmp → rename)
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from si_logging import get_logger

log = get_logger("http")

DEFAULT_UA = "sportintel-core/2.0 (+https://github.com/avramnull/sportintel_core; production)"
DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_READ_TIMEOUT = 90
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_BACKOFF_CAP = 60.0


def _session(
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = 0.8,
) -> requests.Session:
    s = requests.Session()
    # urllib3 Retry handles 429/5xx and connection errors at transport level
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "User-Agent": DEFAULT_UA,
            "Accept": "text/csv,text/plain,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    )
    return s


def download_bytes(
    url: str,
    *,
    timeout: Tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
    max_attempts: int = DEFAULT_MAX_RETRIES + 1,
    min_bytes: int = 100,
    expect_csv: bool = True,
) -> Optional[bytes]:
    """
    Download URL with retries and validation.
    Returns content bytes or None on permanent failure.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            with _session() as sess:
                log.info("GET %s (attempt %d/%d)", url, attempt, max_attempts)
                r = sess.get(url, timeout=timeout, stream=False)
                if r.status_code == 404:
                    log.error("404 Not Found: %s", url)
                    return None
                if r.status_code >= 400:
                    log.warning("HTTP %s for %s", r.status_code, url)
                    # Let outer retry continue for 5xx; 4xx other than 404 are soft-fail
                    if 400 <= r.status_code < 500 and r.status_code != 429:
                        log.error("client error %s — giving up", r.status_code)
                        return None
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)

                content = r.content
                if len(content) < min_bytes:
                    raise ValueError(f"response too small ({len(content)} bytes)")
                if expect_csv:
                    head = content[:400].lower()
                    if b"<html" in head or b"<!doctype" in head:
                        raise ValueError("response looks like HTML, not CSV")
                    # Basic CSV smell check
                    if b"," not in content[:2000] and b";" not in content[:2000]:
                        log.warning("response may not be CSV (no commas/semicolons in head)")

                log.info("downloaded %s bytes from %s", f"{len(content):,}", url)
                return content
        except (requests.RequestException, ValueError, OSError) as e:
            last_err = e
            log.warning("attempt %d failed: %s", attempt, e)
            if attempt >= max_attempts:
                break
            # Extra application-level backoff with jitter (urllib3 already did some)
            sleep = min(DEFAULT_BACKOFF_CAP, DEFAULT_BACKOFF_BASE ** attempt) + random.uniform(0, 1.5)
            log.info("backing off %.1fs before retry", sleep)
            time.sleep(sleep)

    log.error("download failed after %d attempts: %s", max_attempts, last_err)
    return None


def download_to_files(
    url: str,
    out_path: Path,
    stable_path: Optional[Path] = None,
    **kwargs,
) -> Optional[Path]:
    """
    Download and write atomically to out_path (and optional stable_path).
    Returns out_path on success, None on failure.
    """
    content = download_bytes(url, **kwargs)
    if content is None:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        tmp.write_bytes(content)
        tmp.replace(out_path)
        if stable_path is not None:
            stable = Path(stable_path)
            stable.parent.mkdir(parents=True, exist_ok=True)
            # Second atomic write
            stmp = stable.with_suffix(stable.suffix + ".tmp")
            stmp.write_bytes(content)
            stmp.replace(stable)
        log.info("saved %s (%s bytes)%s", out_path.name, f"{len(content):,}",
                 f" + {stable_path.name}" if stable_path else "")
        return out_path
    except OSError as e:
        log.error("write failed: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return None
