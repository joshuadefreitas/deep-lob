"""
The paper's headline numbers must equal the committed evidence.

This is the enforcement for rule 11. The failure it guards against has occurred
three times in this repository: the artifacts were widened to more generator
paths and the paper kept quoting one. It was caught by a reader every time.

Verified to fail: perturbing a single digit of the main table, and perturbing
the abstract's headline excess, both produce a non-zero exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import check_paper_numbers  # noqa: E402


def test_paper_headline_numbers_match_committed_results() -> None:
    assert check_paper_numbers.main() == 0, "see stdout for the mismatches"
