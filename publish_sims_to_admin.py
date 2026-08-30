#!/usr/bin/env python3
"""
Copy daily_football_data/sims/ → sportintel repo data/sims/ and push.

Requires:
  GITHUB_TOKEN or GH_TOKEN with repo scope on avramnull/sportintel
  Optional: SPORTINTEL_REPO (default avramnull/sportintel)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "daily_football_data" / "sims"
REPO = os.environ.get("SPORTINTEL_REPO", "avramnull/sportintel")
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("SPORTINTEL_TOKEN")


def run(cmd, cwd=None):
    print(">>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def main():
    if not SRC.exists() or not (SRC / "index.json").exists():
        raise SystemExit(f"No sims index at {SRC}/index.json")
    if not TOKEN:
        raise SystemExit("Set GITHUB_TOKEN (or GH_TOKEN / SPORTINTEL_TOKEN) to push admin sims")

    url = f"https://x-access-token:{TOKEN}@github.com/{REPO}.git"
    with tempfile.TemporaryDirectory(prefix="sportintel-") as tmp:
        tmp = Path(tmp)
        run(["git", "clone", "--depth", "1", url, str(tmp / "repo")])
        repo = tmp / "repo"
        dest = repo / "data" / "sims"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(SRC, dest)
        readme = dest / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Fixture simulation lab\n\nDaily `sim.py` reports for Control Room → Sim Lab.\n",
                encoding="utf-8",
            )
        run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], cwd=repo)
        run(["git", "config", "user.name", "github-actions[bot]"], cwd=repo)
        run(["git", "add", "data/sims"], cwd=repo)
        # commit only if changes
        st = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=repo)
        if st.returncode == 0:
            print("No sim changes to publish")
            return
        msg = f"Sim Lab daily reports: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        run(["git", "commit", "-m", msg], cwd=repo)
        run(["git", "push", "origin", "HEAD"], cwd=repo)
        print(f"Published sims → https://github.com/{REPO}/tree/main/data/sims")


if __name__ == "__main__":
    main()
