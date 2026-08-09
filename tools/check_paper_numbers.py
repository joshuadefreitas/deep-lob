"""
The paper must not drift from the evidence.

Three times in this project the artifacts were widened to multiple generator
paths and the paper was left quoting one. Each time it was caught by a reader,
not by the repository. This closes that: it recomputes every headline number
from the committed CSVs and compares it to the number printed in the .tex.

Any mismatch is a failure. Exit code 1.

Configuration span
------------------
Checks: the main results table (9 cells x 3 quantities), the ceiling
verification table (3 horizons x 2 quantities), and the abstract's two headline
excesses. NOT checked: the robustness sweep, the stride table, the normalisation
table, and prose numbers outside those tables.

Run: .venv/bin/python tools/check_paper_numbers.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "docs" / "paper" / "leakage-ceiling.tex"
TOL = 5e-5  # the tex prints 4 decimals; half a unit in the last place


def pooled_study() -> dict[str, tuple[float, ...]]:
    """Per-path means, then the mean and spread across paths."""
    per_path: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)
    with (ROOT / "results" / "study-multiseed" / "runs.csv").open() as f:
        for r in csv.DictReader(f):
            key = (r["protocol"], int(r["horizon"]), int(r["sim_seed"]))
            per_path[key].append((float(r["val_acc"]), float(r["val_majority"])))

    out: dict[str, tuple[float, ...]] = {}
    for proto in ("random_split", "chronological", "purged_embargoed"):
        for h in (5, 10, 20):
            paths = sorted(s for (p, hh, s) in per_path if p == proto and hh == h)
            accs, bases, excs = [], [], []
            for s in paths:
                rows = per_path[(proto, h, s)]
                a = mean(x for x, _ in rows)
                b = mean(y for _, y in rows)
                accs.append(a)
                bases.append(b)
                excs.append(a - b)
            out[f"{proto}|h{h}"] = (
                mean(accs), mean(bases), mean(excs),
                stdev(excs), min(excs), max(excs),
            )
    return out


def pooled_ceiling() -> dict[int, tuple[float, float, float]]:
    """Analytic ceiling, and the oracle averaged over generator paths."""
    per_path: dict[tuple[int, int], list[float]] = defaultdict(list)
    ceil: dict[int, float] = {}
    with (ROOT / "results" / "ceiling" / "ceiling.csv").open() as f:
        for r in csv.DictReader(f):
            if r["protocol"] != "random_split":
                continue
            h = int(r["horizon"])
            per_path[(h, int(r["sim_seed"]))].append(float(r["oracle_twin_acc"]))
            ceil[h] = float(r["analytic_ceiling"])

    out = {}
    for h in sorted(ceil):
        by_path = [mean(v) for (hh, _), v in per_path.items() if hh == h]
        out[h] = (ceil[h], mean(by_path), stdev(by_path))
    return out


def main() -> int:
    tex = TEX.read_text()
    failures: list[str] = []

    def check(label: str, expected: float, printed: float) -> None:
        if abs(expected - printed) > TOL:
            failures.append(
                f"{label}: paper says {printed:.4f}, evidence says {expected:.4f} "
                f"(diff {printed - expected:+.4f})"
            )

    # ---- main results table ------------------------------------------------
    study = pooled_study()
    for proto in ("random_split", "chronological", "purged_embargoed"):
        tex_proto = proto.replace("_", r"\_")
        for h in (5, 10, 20):
            pat = (
                rf"\\texttt\{{{re.escape(tex_proto)}\}} & {h} & "
                r"([\d.]+) & ([\d.]+) & \$\\?m?a?t?h?b?f?\{?([+-][\d.]+)\}?\$"
                r" & ([\d.]+) & \$\[([+-][\d.]+), ([+-][\d.]+)\]\$"
            )
            m = re.search(pat, tex)
            if not m:
                failures.append(f"main table row not found in tex: {proto} h={h}")
                continue
            acc, base, ex, sd, lo, hi = study[f"{proto}|h{h}"]
            for name, want, got in (
                ("acc", acc, float(m.group(1))), ("base", base, float(m.group(2))),
                ("excess", ex, float(m.group(3))), ("sd", sd, float(m.group(4))),
                ("min", lo, float(m.group(5))), ("max", hi, float(m.group(6))),
            ):
                check(f"tab:main {proto} h={h} {name}", want, got)

    # ---- ceiling verification table ---------------------------------------
    for h, (cl, om, osd) in pooled_ceiling().items():
        m = re.search(
            rf"^{h} & [\d.]+ & ([\d.]+) & ([\d.]+) \$\\pm\$ ([\d.]+) &",
            tex, re.MULTILINE,
        )
        if not m:
            failures.append(f"ceiling table row not found in tex: h={h}")
            continue
        check(f"ceiling h={h} C(h)", cl, float(m.group(1)))
        check(f"ceiling h={h} oracle", om, float(m.group(2)))
        check(f"ceiling h={h} sd", osd, float(m.group(3)))

    # ---- abstract headline excesses (printed to 1 decimal, in points) ------
    abstract = tex[tex.index(r"\begin{abstract}"):tex.index(r"\end{abstract}")]
    printed = [float(x) for x in re.findall(r"\$([+-]\d+\.\d)\$ points", abstract)]
    want = [study["random_split|h20"][2] * 100, study["purged_embargoed|h20"][2] * 100]
    if len(printed) < 2:
        failures.append("abstract: could not find two headline excesses")
    else:
        for w, p in zip(want, printed[:2]):
            if abs(w - p) > 0.05:
                failures.append(f"abstract headline: paper {p:+.1f}, evidence {w:+.1f}")

    n_runs = sum(1 for _ in csv.DictReader((ROOT / "results" / "study-multiseed" / "runs.csv").open()))
    if rf"${n_runs}$ runs" not in tex:
        failures.append(f"run count: evidence has {n_runs} runs; tex does not say so")

    if failures:
        print("PAPER DOES NOT MATCH EVIDENCE\n")
        for f in failures:
            print("  FAIL", f)
        print(f"\n{len(failures)} mismatch(es)")
        return 1

    print(f"paper matches evidence: 9 table cells x 6, 3 ceiling rows x 3, "
          f"2 abstract headlines, {n_runs} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
