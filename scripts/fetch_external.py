"""Download the external benchmark datasets into external/.

IDSSE (DFL Bundesliga 2022/23, CC BY 4.0) comes from its figshare record and every file is checked
against the md5 figshare publishes. SkillCorner Open Data (MIT) is pinned to one commit so a later
push to that repository cannot change the benchmark underneath us.

    python scripts/fetch_external.py                 # both
    python scripts/fetch_external.py --only idsse
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(ROOT, "external")

IDSSE_ARTICLE = 28196177
SKILLCORNER_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
SKILLCORNER_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/".format(
    SKILLCORNER_COMMIT)
# Tracking files are stored with git LFS, so they come from the media host instead of raw.
SKILLCORNER_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/data/".format(
    SKILLCORNER_COMMIT)


def _get(url, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _download(url, path, md5=None):
    if os.path.isfile(path) and (md5 is None or _md5(path) == md5):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if md5 is not None and _md5(tmp) != md5:
        os.remove(tmp)
        raise IOError("checksum mismatch for {}".format(url))
    os.replace(tmp, path)
    return True


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_idsse():
    record = json.loads(_get("https://api.figshare.com/v2/articles/{}".format(IDSSE_ARTICLE)))
    target = os.path.join(EXTERNAL, "idsse")
    for item in record["files"]:
        path = os.path.join(target, item["name"])
        fresh = _download(item["download_url"], path, item.get("computed_md5") or None)
        print("{} {}".format("got " if fresh else "have", item["name"]))
    with open(os.path.join(target, "LICENSE.txt"), "w") as handle:
        handle.write("{}\n{}\ndoi:{}\n".format(record["title"], record["license"]["url"],
                                               record["doi"]))


def fetch_skillcorner():
    target = os.path.join(EXTERNAL, "skillcorner")
    matches = json.loads(_get(SKILLCORNER_RAW + "matches.json"))
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "matches.json"), "w") as handle:
        json.dump(matches, handle)
    for match in matches:
        mid = str(match["id"])
        for suffix, base in (("_match.json", SKILLCORNER_RAW),
                             ("_phases_of_play.csv", SKILLCORNER_RAW),
                             ("_dynamic_events.csv", SKILLCORNER_RAW),
                             ("_tracking_extrapolated.jsonl", SKILLCORNER_LFS)):
            name = mid + suffix
            fresh = _download(base + "matches/{}/{}".format(mid, name),
                              os.path.join(target, mid, name))
            print("{} {}".format("got " if fresh else "have", name))
    with open(os.path.join(target, "LICENSE.txt"), "wb") as handle:
        handle.write(_get("https://raw.githubusercontent.com/SkillCorner/opendata/{}/LICENSE"
                          .format(SKILLCORNER_COMMIT)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=("idsse", "skillcorner"))
    args = parser.parse_args()
    if args.only in (None, "skillcorner"):
        fetch_skillcorner()
    if args.only in (None, "idsse"):
        fetch_idsse()
    sys.exit(0)
