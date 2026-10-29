"""Download the original Steam Games CSV from Zenodo."""
from pathlib import Path
import hashlib
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/15121027/files/steam-games.csv?download=1"
DEST = ROOT / "data" / "raw" / "steam-games.csv"
EXPECTED_MD5 = "a101d45e1acb048dc09f35680130a2a7"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset from Zenodo...")
    response = requests.get(URL, timeout=90)
    response.raise_for_status()
    content = response.content
    if len(content) < 1000 or b"<!doctype html" in content[:500].lower():
        raise RuntimeError("The downloaded response does not look like the expected CSV file.")
    digest = hashlib.md5(content).hexdigest()
    DEST.write_bytes(content)
    print(f"Saved: {DEST}")
    print(f"Size: {len(content):,} bytes")
    print(f"MD5: {digest}")
    if digest != EXPECTED_MD5:
        print("Warning: checksum differs from the checksum shown on Zenodo. Inspect the downloaded file before training.")
    else:
        print("Checksum matches the Zenodo record.")


if __name__ == "__main__":
    main()
