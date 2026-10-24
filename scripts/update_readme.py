"""Copy generated result tables into README.md so every number there comes from a script.

README.md contains marker pairs like
    <!-- BEGIN results/retrieval/bm25_page.md -->
    ...
    <!-- END results/retrieval/bm25_page.md -->
and this script replaces what is between them with the current file content.
A missing results file leaves a [not yet measured] placeholder.
"""
import re
from pathlib import Path

README = Path("README.md")
BLOCK = re.compile(r"(<!-- BEGIN (\S+) -->\n)(.*?)(<!-- END \2 -->)", re.DOTALL)


def fill(match: re.Match) -> str:
    path = Path(match.group(2))
    body = path.read_text() if path.exists() else f"[not yet measured: run the script that writes `{path}`]\n"
    return match.group(1) + body + match.group(4)


def main() -> None:
    README.write_text(BLOCK.sub(fill, README.read_text()))
    print("README tables refreshed")


if __name__ == "__main__":
    main()
