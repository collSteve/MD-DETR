import argparse
import re
from pathlib import Path

# Regex: split on the first <, >, =, or ~ that begins a version/operator
_SPLIT_PATTERN = re.compile(r"[<>=~]", re.ASCII)

def strip_line(line: str) -> str | None:
    """
    Return the package name without version specifier.
    Ignores comments, blank lines, and `-r otherfile` includes.
    """
    line = line.strip()

    # Ignore comments and empty lines
    if not line or line.startswith("#"):
        return None

    # Ignore recursive requirements includes
    if line.lower().startswith(("-r", "--requirement")):
        return None

    # Remove inline comment (everything after an unescaped '#')
    if "#" in line:
        line = re.split(r"(?<!\\)#", line, maxsplit=1)[0].strip()

    # Split on first operator char to isolate package name
    pkg = _SPLIT_PATTERN.split(line, maxsplit=1)[0].strip()

    # Handle extras like pkg[extra1,extra2]==1.2.3  ->  pkg[extra1,extra2]
    return pkg if pkg else None


def main(src: Path, dst: Path) -> None:
    packages: list[str] = []

    with src.open("r", encoding="utf-8") as f:
        for raw_line in f:
            pkg = strip_line(raw_line)
            if pkg:
                packages.append(pkg)

    dst.write_text("\n".join(packages) + "\n", encoding="utf-8")
    print(f"Wrote {len(packages)} package names to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strip version specifiers from a requirements.txt file."
    )
    parser.add_argument(
        "input", type=Path, help="Path to the original requirements file"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("requirements_no_version.txt"),
        help="Path for the cleaned output file (default: requirements_no_version.txt)",
    )
    args = parser.parse_args()
    main(args.input, args.output)