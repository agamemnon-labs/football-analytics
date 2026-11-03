"""Print the structure of the match bundle without printing any of its content.

Every line this script writes is a name, a count, a fraction or a min/max over a whole column.
No row, no per-frame value and no coordinate of any single object is ever shown, so the output
can be shared and discussed freely.

    python scripts/inspect_schema.py
"""

import collections
import csv
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402

# Only these elements hold channel and format definitions (schema, not measurements), so their
# attribute values are printed. Everything else is reported by tag and attribute name only.
SCHEMA_TAGS = {"DataFormatSpecification", "StringRegister", "SplitRegister", "PlayerChannelRef",
               "BallChannelRef", "PlayerChannel", "BallChannel", "Channel", "DataSpecification"}


def xml_tree(path):
    root = ET.parse(path).getroot()
    paths = collections.Counter()
    attrs = collections.defaultdict(set)
    schema = []

    def walk(node, prefix):
        here = prefix + "/" + node.tag
        paths[here] += 1
        attrs[here].update(node.attrib)
        if node.tag in SCHEMA_TAGS:
            schema.append((here, dict(node.attrib)))
        for child in node:
            walk(child, here)

    walk(root, "")
    print("  tag paths (count, attribute names):")
    for here in sorted(paths):
        print("    {:>6}  {}  {}".format(paths[here], here, sorted(attrs[here])))
    if schema:
        print("  schema elements:")
        # Per-slot channel definitions repeat 46 times; print each shape once with a count.
        shapes = collections.Counter()
        first = {}
        for here, attrib in schema:
            shape = (here, tuple(sorted((k, re.sub(r"\d+", "#", v)) for k, v in attrib.items())))
            shapes[shape] += 1
            first.setdefault(shape, attrib)
        for shape, n in shapes.items():
            print("    x{:<4} {}  {}".format(n, shape[0], dict(shape[1])))
    return root


def atd_meta(path):
    print("\n== ATD metadata: {}".format(os.path.basename(path)))
    root = xml_tree(path)
    teams = [team.get("id") for team in root.iter("Team")]
    names = [team.findtext("Name") for team in root.iter("Team")]
    per_team = collections.Counter(player.get("teamId") for player in root.iter("Player"))
    print("  teams in file order: {}".format(names))
    print("  player slots per team (file order): {}".format([per_team[t] for t in teams]))


def atd_raw(path):
    print("\n== ATD raw: {}".format(os.path.basename(path)))
    n_lines = 0
    n_parts = collections.Counter()
    n_groups = collections.Counter()
    player_fields = collections.Counter()
    ball_fields = collections.Counter()
    first = last = None
    contiguous = True
    ball_stats = None
    player_stats = None

    def update(stats, values):
        for k, v in enumerate(values):
            if v in ("NaN", "nan", ""):
                stats[k][0] += 1
                continue
            try:
                f = float(v)
            except ValueError:
                stats[k][3] += 1
                continue
            if math.isnan(f):
                stats[k][0] += 1
                continue
            stats[k][1] = min(stats[k][1], f)
            stats[k][2] = max(stats[k][2], f)

    with open(path) as handle:
        for line in handle:
            n_lines += 1
            parts = line.rstrip("\n").split(":")
            n_parts[len(parts)] += 1
            try:
                frame_id = int(parts[0])
            except ValueError:
                print("  line {}: frame id not an integer".format(n_lines))
                continue
            if last is not None and frame_id != last + 1:
                contiguous = False
            first = frame_id if first is None else first
            last = frame_id
            if len(parts) < 3:
                continue
            groups = parts[1].split(";")
            n_groups[len(groups)] += 1
            for group in groups:
                fields = group.split(",")
                player_fields[len(fields)] += 1
                if player_stats is None:
                    player_stats = [[0, math.inf, -math.inf, 0] for _ in fields]
                if len(fields) == len(player_stats):
                    update(player_stats, fields)
            ball = parts[2].split(",")
            ball_fields[len(ball)] += 1
            if ball_stats is None:
                ball_stats = [[0, math.inf, -math.inf, 0] for _ in ball]
            if len(ball) == len(ball_stats):
                update(ball_stats, ball)

    print("  lines {}  frame_id {}..{}  contiguous {}".format(n_lines, first, last, contiguous))
    print("  ':'-parts per line: {}".format(dict(n_parts)))
    print("  player groups per line: {}".format(dict(n_groups)))
    print("  fields per player group: {}".format(dict(player_fields)))
    print("  fields per ball group: {}".format(dict(ball_fields)))
    for label, stats, total in (("player", player_stats, sum(player_fields.values())),
                                ("ball", ball_stats, sum(ball_fields.values()))):
        for k, (n_nan, lo, hi, n_text) in enumerate(stats or []):
            print("  {} field {}: missing {:.1%}  non-numeric {}  min {:.4g}  max {:.4g}".format(
                label, k, n_nan / max(total, 1), n_text, lo, hi))


def smart_tagging(path):
    print("\n== Smart Tagging: {}".format(os.path.basename(path)))
    root = xml_tree(path)
    codes = collections.Counter(inst.findtext("code") for inst in root.iter("instance"))
    groups = collections.defaultdict(collections.Counter)
    for inst in root.iter("instance"):
        for label in inst.findall("label"):
            groups[label.findtext("group")][label.findtext("text")] += 1
    print("  instances {}".format(sum(codes.values())))
    for code, n in codes.most_common():
        print("    {:<34} {}".format(code, n))
    for group in sorted(groups, key=str):
        values = groups[group]
        if group in ("Team", "Type", "Half"):
            print("  group {}: {}".format(group, dict(values)))
        else:
            print("  group {}: {} distinct values, {} labels".format(
                group, len(values), sum(values.values())))


def homography(path):
    print("\n== Homography: {}".format(os.path.basename(path)))
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = valid = 0
        for row in reader:
            rows += 1
            valid += row[-1].strip() == "1"
    print("  columns {}".format(header))
    print("  rows {}  valid {:.1%}".format(rows, valid / max(rows, 1)))


if __name__ == "__main__":
    for source in ("tactical", "broadcast"):
        paths = asset_paths(source=source)
        print("\n######## {} : files present {}".format(
            source, {k: os.path.isfile(v) for k, v in paths.items()}))
    paths = asset_paths()
    atd_meta(paths["atd_meta"])
    atd_raw(paths["atd_raw"])
    smart_tagging(paths["smart_tagging"])
    homography(paths["homography"])
