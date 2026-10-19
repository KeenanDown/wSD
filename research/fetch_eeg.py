"""Fetch a fixed, outcome-independent EEG pilot and verify upstream SHA256s."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://physionet.org/files/eegmmidb/1.0.0/"
RAW = ROOT / "data/raw"
RAW.mkdir(parents=True, exist_ok=True)


def fetch_one(item):
    relative, expected = item
    path = RAW / Path(relative).name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        tmp = path.with_suffix(".part")
        with urlopen(BASE + relative, timeout=300) as response, tmp.open("wb") as out:
            while chunk := response.read(1024*128):
                out.write(chunk)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"SHA256 mismatch: {relative}")
        tmp.replace(path)
    print(f"verified {relative}", flush=True)
    return {"record": path.stem, "url": BASE + relative, "sha256": expected,
            "bytes": path.stat().st_size}


if __name__ == "__main__":
    with urlopen(BASE + "SHA256SUMS.txt", timeout=120) as response:
        checksums = response.read().decode()
    (RAW / "upstream-SHA256SUMS.txt").write_text(checksums)
    lookup = {line.split()[1].lstrip("./"): line.split()[0] for line in checksums.splitlines() if line.strip()}
    # First five subjects, both baseline conditions; fixed before outcomes.
    relative = [f"S{s:03d}/S{s:03d}R{r:02d}.edf" for s in range(1, 6) for r in (1, 2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch_one, [(p, lookup[p]) for p in relative]))
    manifest = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0",
        "doi": "10.13026/C28G6P",
        "license": "Open Data Commons Attribution License v1.0",
        "license_url": "https://physionet.org/content/eegmmidb/view-license/1.0.0/",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "S001-S005, R01 eyes open and R02 eyes closed, selected before analysis",
        "records": records,
    }
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
