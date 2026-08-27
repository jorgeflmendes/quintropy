"""Validate checksums of frozen research artefacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative_path = line.split("  ", maxsplit=1)
        path = ROOT / relative_path
        actual = sha256(path)
        if actual != expected:
            failures.append(relative_path)

    if failures:
        raise SystemExit(f"Checksum mismatch: {', '.join(failures)}")
    print("All frozen artefact checksums match.")


if __name__ == "__main__":
    main()
