#!/usr/bin/env python3
"""Offline integrity checks for the public recipe."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
runtime = root / "runtime"
manifest = json.loads((runtime / "patch_manifest.json").read_text(encoding="utf-8"))

assert len(manifest["patches"]) == 7

for item in manifest["patches"]:
    path = runtime / item["file"]
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path

for item in manifest["overlay_files"]:
    path = runtime / item["path"]
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path

assert not list(root.rglob(".git"))
assert all(p.stat().st_size <= 100_000_000 for p in root.rglob("*") if p.is_file())

for forbidden in ("*.safetensors", "*.gguf", "*.pem", "*.p12", "*.pfx"):
    assert not list(root.rglob(forbidden)), forbidden

print("PASS: patch hashes, overlay hashes, file-size bound, no nested Git history or bundled weights/keys")
