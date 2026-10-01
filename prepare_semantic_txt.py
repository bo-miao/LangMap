"""Convert official HM3D-Sem <name>.semantic.txt files to the category names used by LangMap.

LangMap normalizes a few HM3D-Sem category names (e.g., "trash can" -> "trashcan"). Instead of redistributing
HM3D-Sem files, this script rewrites the official files in place and keeps the original as <file>.orig.

    python prepare_semantic_txt.py --scene-dir data/hm3d/val
"""
import argparse
import glob
import os
import shutil

RENAME = {
    "trash can": "trashcan",
    "statue/art": "statue",
    "shoes": "shoe",
    "ceiling light": "ceiling lamp",
    "small table/stand": "table",
    "cabinet clutter": "cabinet",
    "shelf / cabinet": "unknown",
    "kitchen utensils": "kitchen utensil",
}


def convert_line(line):
    """Lines look like: 12,A1B2C3,"trash can",4 — rename the quoted category, keep everything else unchanged."""
    parts = line.split(",", 2)
    if len(parts) < 3 or not parts[2].startswith('"'):
        return line
    end = parts[2].find('"', 1)
    if end < 0 or parts[2][end + 1:end + 2] != ",":  # malformed or escaped quote inside the name: leave unchanged
        return line
    name = parts[2][1:end]
    if name not in RENAME:
        return line
    return f'{parts[0]},{parts[1]},"{RENAME[name]}"{parts[2][end + 1:]}'


def convert(text):
    lines = text.split("\n")
    return "\n".join(lines[:1] + [convert_line(line) for line in lines[1:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="data/hm3d/val", help="folder with <id>-<name>/<name>.semantic.txt")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.scene_dir, "*", "*.semantic.txt")))
    assert files, f"no *.semantic.txt under {args.scene_dir}"
    changed = 0
    for path in files:
        with open(path, encoding="utf-8", newline="") as f:  # keep line endings unchanged
            text = f.read()
        new = convert(text)
        if new != text:
            if not os.path.exists(path + ".orig"):
                shutil.copy2(path, path + ".orig")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            changed += 1
    print(f"{len(files)} semantic.txt files checked, {changed} rewritten")


if __name__ == "__main__":
    main()
