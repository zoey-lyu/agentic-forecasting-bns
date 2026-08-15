"""Download and cache official public Scotiabank PDFs for the ALCO prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "implementations/boc_rate_decisions/scotiabank_alco_manifest.json"
CACHE_DIR = REPO_ROOT / "data/reports/scotiabank_alco"
USER_AGENT = "Mozilla/5.0 (compatible; agentic-forecasting/1.0; +public-document-cache)"


def main() -> None:
    """Download manifest documents and write provenance sidecars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "provenance").mkdir(exist_ok=True)

    for entry in manifest["documents"]:
        path = CACHE_DIR / f"{entry['doc_id']}.pdf"
        if path.exists() and not args.force:
            print(f"skip (cached): {path}")
            continue
        request = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - allowlisted manifest URLs
            body = response.read()
            status = int(getattr(response, "status", 200) or 200)
        if not body.startswith(b"%PDF"):
            raise RuntimeError(f"Non-PDF response for {entry['url']}")
        path.write_bytes(body)
        provenance = {
            **entry,
            "http_status": status,
            "retrieved_at": datetime.now(tz=timezone.utc).isoformat(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "content_length": len(body),
        }
        sidecar = CACHE_DIR / "provenance" / f"{entry['doc_id']}.json"
        sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"ok: {path} ({len(body):,} bytes)")


if __name__ == "__main__":
    main()