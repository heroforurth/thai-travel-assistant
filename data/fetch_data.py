"""Download the corpus: plain-text Wikivoyage guides for 15 Thai destinations.

Run:  python data/fetch_data.py        (writes data/raw/<Destination>.txt)

Wikivoyage text is CC BY-SA 4.0 — see data/raw/SOURCES.md for attribution.
"""
import json
import time
import pathlib
import urllib.parse
import urllib.error
import urllib.request

DESTINATIONS = [
    "Ayutthaya", "Chiang Mai", "Chiang Rai", "Pai", "Sukhothai",
    "Kanchanaburi", "Lopburi", "Hua Hin", "Ao Nang", "Ko Tao",
    "Ko Samet", "Ko Lanta", "Phuket", "Khao Yai National Park", "Nan",
]
API = "https://en.wikivoyage.org/w/api.php"
OUT = pathlib.Path(__file__).parent / "raw"


def fetch(title: str) -> tuple[str, int]:
    params = {"action": "query", "prop": "extracts|revisions", "explaintext": 1, "rvprop": "ids",
              "titles": title, "format": "json", "formatversion": 2, "redirects": 1}
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": "kmitl-capstone-student/1.0"})
    for attempt in range(5):  # Wikivoyage rate-limits bursts (HTTP 429): back off and retry
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                page = json.load(r)["query"]["pages"][0]
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
    return page["extract"], page["revisions"][0]["revid"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Sources (Wikivoyage, CC BY-SA 4.0)\n"]
    for title in DESTINATIONS:
        text, revid = fetch(title)
        (OUT / f"{title}.txt").write_text(text, encoding="utf-8")
        url = f"https://en.wikivoyage.org/w/index.php?title={urllib.parse.quote(title)}&oldid={revid}"
        lines.append(f"- {title}: {url}")
        time.sleep(3)
        print(f"{title:25s} {len(text):>7d} chars  rev {revid}")
    (OUT / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
