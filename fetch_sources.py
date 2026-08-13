"""Download the OpenStax Calculus volumes into sources/.

The PDFs are ~170 MB total, so they are gitignored rather than committed.
OpenStax Calculus is CC BY-NC-SA 4.0: https://openstax.org/details/books/calculus-volume-1
"""

from pathlib import Path
from urllib.request import urlopen

SOURCES = Path("sources")
BASE = "https://assets.openstax.org/oscms-prodcms/media/documents"
VOLUMES = {
    f"openstax_calculus_volume_{v}.pdf": f"{BASE}/calculus-volume-{v}_-_WEB.pdf"
    for v in (1, 2, 3)
}


def main() -> None:
    SOURCES.mkdir(exist_ok=True)
    for name, url in VOLUMES.items():
        target = SOURCES / name
        if target.exists():
            print(f"skip  {name} (already present)")
            continue
        print(f"fetch {name} ...", flush=True)
        with urlopen(url) as response:
            target.write_bytes(response.read())
        print(f"      {target.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
