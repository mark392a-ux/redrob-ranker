import gzip
import json
from pathlib import Path


def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    candidates = []
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_jd_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
