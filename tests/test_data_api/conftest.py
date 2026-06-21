from __future__ import annotations

import difflib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest


def extract_json_code_blocks(
    markdown_content: str,
) -> dict[tuple[int, int], dict[str, object]]:
    """
    Extract all JSON code blocks from a markdown document.

    Each extracted code block includes the JSON content and its position in the document.

    Args:
        markdown_content: The markdown document content as a string.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - 'json': The parsed JSON object (or None if parsing failed)
        - 'raw': The raw JSON string
        - 'start_line': The line number where the code block starts (1-indexed)
        - 'end_line': The line number where the code block ends (1-indexed)
        - 'error': Error message if JSON parsing failed (optional)
    """
    lines: list[str] = markdown_content.split("\n")
    code_blocks: dict[tuple[int, int], dict[str, object]] = {}
    i: int = 0

    while i < len(lines):
        line: str = lines[i]

        # Check for JSON code block start
        if re.match(r"^\s*```json\s*$", line):
            start_line: int = i + 1  # Line after ```json
            json_lines: list[str] = []
            i += 1

            # Collect lines until we find the closing ```
            while i < len(lines) and not re.match(r"^\s*```\s*$", lines[i]):
                json_lines.append(lines[i])
                i += 1

            end_line: int = i  # Line with closing ```
            raw_json: str = "\n".join(json_lines)

            parsed_json: object = json.loads(raw_json)
            assert isinstance(parsed_json, dict)
            code_blocks[(start_line + 1, end_line)] = parsed_json

        i += 1

    return code_blocks


@pytest.fixture
def proj_attrs_examples() -> dict[tuple[int, int], dict[str, object]]:
    """
    Extract JSON code blocks from the proj extension README.md.
    """
    spec_md = Path(__file__).parent.parent.parent / "attributes" / "geo" / "proj" / "README.md"
    content = spec_md.read_text(encoding="utf-8")
    return extract_json_code_blocks(content)


@pytest.fixture
def json_code_blocks_from_readme(
    request: pytest.FixtureRequest,
) -> dict[tuple[int, int], dict[str, object]]:
    """Extract JSON code blocks from a markdown file."""
    readme_path = Path(request.param)
    content = readme_path.read_text(encoding="utf-8")
    return extract_json_code_blocks(content)


def view_json_diff(
    a: dict[str, object],
    b: dict[str, object],
    *,
    sort_keys: bool = True,
    indent: int = 2,
) -> str:
    """
    Generate a human-readable diff between two JSON objects
    """
    a_str = json.dumps(a, indent=indent, sort_keys=sort_keys)
    b_str = json.dumps(b, indent=indent, sort_keys=sort_keys)

    # difflib.unified_diff returns an iterable of lines
    diff = difflib.unified_diff(
        a_str.splitlines(keepends=True),
        b_str.splitlines(keepends=True),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    return "".join(diff)


def json_eq(a: object, b: object) -> bool:
    """
    An equality check between python objects that recurses into dicts and sequences and ignores
    the difference between tuples and lists. Otherwise, it's just regular equality. Useful
    for comparing dicts that would become identical JSON, but where one has lists and the other
    has tuples.
    """
    # treat lists & tuples as the same "sequence" type
    seq_types = (list, tuple)

    # both are sequences → compare element-wise
    if isinstance(a, seq_types) and isinstance(b, seq_types):
        return len(a) == len(b) and all(json_eq(x, y) for x, y in zip(a, b, strict=False))

    # recurse into mappings
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(json_eq(a[k], b[k]) for k in a)

    # otherwise → regular equality
    return a == b
