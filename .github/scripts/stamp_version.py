import re
import sys
from pathlib import Path

TAG_RE = re.compile(r"v(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)")


def main(tag: str) -> None:
    match = TAG_RE.fullmatch(tag)
    if match is None:
        sys.exit(f"release tag {tag!r} must look like v1.2.3 or v1.2.3rc1")
    version = match.group(1)

    pyproject = Path("pyproject.toml")
    stamped, count = re.subn(
        r'^version = "0\.0\.0"$',
        f'version = "{version}"',
        pyproject.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        sys.exit('expected exactly one version = "0.0.0" line in pyproject.toml')
    pyproject.write_text(stamped)
    print(f"stamped {version} from tag {tag}")


if __name__ == "__main__":
    main(sys.argv[1])
