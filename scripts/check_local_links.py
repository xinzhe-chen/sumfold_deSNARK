#!/usr/bin/env python3
"""Validate local Markdown links used in repository-facing docs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
IGNORED_PREFIXES = ("http://", "https://", "mailto:")


def check_file(markdown_path: Path) -> list[str]:
    errors: list[str] = []
    content = markdown_path.read_text(encoding="utf-8")
    for raw_target in LINK_RE.findall(content):
        target = raw_target.strip()
        if not target or target.startswith(IGNORED_PREFIXES) or target.startswith("#"):
            continue

        path_part = target.split("#", 1)[0]
        if not path_part:
            continue

        resolved = (markdown_path.parent / path_part).resolve()
        if not resolved.exists():
            errors.append(f"{markdown_path}: missing link target {target}")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_local_links.py <markdown-file> [<markdown-file> ...]", file=sys.stderr)
        return 2

    errors: list[str] = []
    for arg in argv[1:]:
        path = Path(arg)
        if not path.exists():
            errors.append(f"{path}: file does not exist")
            continue
        errors.extend(check_file(path))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {len(argv) - 1} markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
