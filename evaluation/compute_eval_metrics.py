#!/usr/bin/env python3
"""
evaluation/compute_eval_metrics.py

Compute manuscript-ready evaluation metrics from a completed (or in-progress)
eval run.

Reads:
    <output_dir>/per_episode.jsonl
    <output_dir>/per_step.jsonl
    <output_dir>/eval_summary.json   (if present, used only as a sanity check)

Computes:
    - Success rate (dissociated / total)
    - <Ea_eff> over dissociated episodes  (eV, T=1000 K)
    - Median Ea_eff
    - <R_Ea> over dissociated episodes
    - Step-level Ea distribution
    - MODE vs SAMPLE breakdown
    - Per-step wall-time stats
    - Comparison-ready table for manuscript training cycles

Reconstruction:
    Ea_eff is read directly from per_episode.jsonl when present. As a
    cross-check, we also recompute it from per_step.jsonl using the same
    formula as realize/ppo/rewards.py:

        Ea_eff = k_B T * ln( Sigma_i exp(E_i / (k_B T)) )

    Only valid steps with ea >= 0 are included. Discrepancies between the
    stored Ea_eff and the recomputed value indicate either:
      (a) the per_step record had a step counted as valid but with no NEB
          barrier reported (rare; would mean ea = None for a valid step), OR
      (b) the per_episode record was written with a stale/incomplete step list.

    Both should be ~zero in a clean run.

Usage:
    python -m evaluation.compute_eval_metrics --output-dir eval_runs/ep442
    python -m evaluation.compute_eval_metrics --output-dir eval_runs/ep442 --markdown
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# Boltzmann constant in eV/K (matches realize/ppo/rewards.py).
K_B = 8.617333e-5
T_KELVIN = 1000.0
KT = K_B * T_KELVIN  # ~= 0.0862 eV at 1000 K


def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] skipped malformed JSON line in {path}: {e}",
                      file=sys.stderr)
    return out


def recompute_ea_eff(step_eas_eV: List[float]) -> Optional[float]:
    """
    Effective activation energy from per-step NEB barriers, with the same
    log-sum-exp form used at training time.

        Ea_eff = k_B T * ln( Sigma_i exp(E_i / k_B T) )

    Numerically stable via log-sum-exp shift.
    Returns None if no valid step Ea is available.
    """
    if not step_eas_eV:
        return None
    arr = [e for e in step_eas_eV if e is not None and e >= 0.0]
    if not arr:
        return None
    e_max = max(arr)
    s = sum(math.exp((e - e_max) / KT) for e in arr)
    return e_max + KT * math.log(s)


def _safe_mean(xs: List[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def _safe_median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _fmt(x: Optional[float], spec: str = ".3f") -> str:
    return "--" if x is None else format(x, spec)


def _percentiles(xs: List[float], pcts: Tuple[int, ...] = (10, 25, 50, 75, 90)) -> Dict[int, float]:
    if not xs:
        return {}
    s = sorted(xs)
    out = {}
    for p in pcts:
        # Linear interpolation; matches numpy.percentile default 'linear'.
        rank = (p / 100.0) * (len(s) - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            out[p] = s[lo]
        else:
            out[p] = s[lo] + (s[hi] - s[lo]) * (rank - lo)
    return out


def analyze(output_dir: str) -> Dict:
    """Build the full metrics dict for one run directory."""
    per_episode_path = os.path.join(output_dir, "per_episode.jsonl")
    per_step_path = os.path.join(output_dir, "per_step.jsonl")
    summary_path = os.path.join(output_dir, "eval_summary.json")

    episodes = _load_jsonl(per_episode_path)
    steps = _load_jsonl(per_step_path)

    # Group steps by episode.
    steps_by_ep: Dict[int, List[Dict]] = defaultdict(list)
    for s in steps:
        steps_by_ep[s.get("episode", -1)].append(s)
    for ep_id in steps_by_ep:
        steps_by_ep[ep_id].sort(key=lambda r: r.get("step", 0))

    # ---- Episode-level aggregates (use stored values; only recompute as a check). ----
    n_total = len(episodes)
    n_dissociated = sum(1 for e in episodes if e.get("dissociated"))
    diss_eps = [e for e in episodes if e.get("dissociated")]

    # Outcome breakdown.
    outcome_counts: Dict[str, int] = defaultdict(int)
    for e in episodes:
        outcome_counts[e.get("outcome", "unknown")] += 1

    ea_eff_list = [e["ea_eff"] for e in diss_eps if e.get("ea_eff") is not None]
    R_Ea_list = [e["R_Ea"] for e in diss_eps if e.get("R_Ea") is not None]
    R_tot_list = [e["R_tot"] for e in diss_eps if e.get("R_tot") is not None]
    steps_list = [e["n_steps"] for e in diss_eps if e.get("n_steps") is not None]
    neb_fail_list = [e["n_neb_failures"] for e in diss_eps if e.get("n_neb_failures") is not None]

    # ---- Recompute Ea_eff per dissociated episode and compare. ----
    ea_eff_recompute: List[Tuple[int, Optional[float], Optional[float]]] = []
    for e in diss_eps:
        ep_id = e["episode"]
        step_eas = [
            s["ea"] for s in steps_by_ep.get(ep_id, [])
            if s.get("valid") and s.get("ea") is not None and s["ea"] >= 0.0
        ]
        recomp = recompute_ea_eff(step_eas)
        ea_eff_recompute.append((ep_id, e.get("ea_eff"), recomp))

    # Discrepancies between stored vs recomputed (informational only).
    discrepancies = [
        (ep, stored, recomp)
        for ep, stored, recomp in ea_eff_recompute
        if (stored is not None and recomp is not None
            and abs(stored - recomp) > 1e-3)
    ]

    # ---- Step-level stats ----
    valid_eas = [s["ea"] for s in steps if s.get("valid") and s.get("ea") is not None and s["ea"] >= 0.0]
    step_times = [s["step_time_s"] for s in steps if s.get("step_time_s") is not None]

    mode_valid = sum(1 for s in steps if s.get("deterministic") and s.get("valid"))
    mode_invalid = sum(1 for s in steps if s.get("deterministic") and not s.get("valid"))
    sample_valid = sum(1 for s in steps if not s.get("deterministic", True) and s.get("valid"))
    sample_invalid = sum(1 for s in steps if not s.get("deterministic", True) and not s.get("valid"))

    # Per-episode steps-to-dissociation distribution.
    n_steps_diss = [len(steps_by_ep[e["episode"]]) for e in diss_eps if e["episode"] in steps_by_ep]

    # kappa distribution (sanity-check the trained vMF concentration on this run).
    log_kappas = [s["log_kappa"] for s in steps if s.get("log_kappa") is not None]
    kappas = [math.exp(lk) for lk in log_kappas]

    return {
        "output_dir": output_dir,
        "n_total": n_total,
        "n_dissociated": n_dissociated,
        "success_rate": n_dissociated / n_total if n_total > 0 else None,
        "outcome_counts": dict(outcome_counts),

        # Manuscript headline numbers.
        "ea_eff_mean": _safe_mean(ea_eff_list),
        "ea_eff_median": _safe_median(ea_eff_list),
        "ea_eff_std": statistics.stdev(ea_eff_list) if len(ea_eff_list) > 1 else None,
        "ea_eff_pcts": _percentiles(ea_eff_list),
        "R_Ea_mean": _safe_mean(R_Ea_list),
        "R_Ea_median": _safe_median(R_Ea_list),
        "R_tot_mean": _safe_mean(R_tot_list),
        "steps_to_diss_mean": _safe_mean([float(x) for x in steps_list]),
        "steps_to_diss_median": _safe_median([float(x) for x in steps_list]),
        "neb_fails_per_diss_mean": _safe_mean([float(x) for x in neb_fail_list]),

        # Cross-check.
        "ea_eff_discrepancies": discrepancies,

        # Step-level.
        "n_steps_total": len(steps),
        "n_valid_steps": sum(1 for s in steps if s.get("valid")),
        "step_ea_mean": _safe_mean(valid_eas),
        "step_ea_median": _safe_median(valid_eas),
        "step_ea_pcts": _percentiles(valid_eas),
        "step_time_mean": _safe_mean(step_times),
        "step_time_median": _safe_median(step_times),
        "step_time_max": max(step_times) if step_times else None,
        "total_step_walltime_h": sum(step_times) / 3600.0 if step_times else None,

        # Mode/sample.
        "mode_valid": mode_valid,
        "mode_invalid": mode_invalid,
        "sample_valid": sample_valid,
        "sample_invalid": sample_invalid,
        "mode_success_rate": mode_valid / max(1, mode_valid + mode_invalid),
        "sample_success_rate": sample_valid / max(1, sample_valid + sample_invalid),

        # Distributions.
        "kappa_mean": _safe_mean(kappas),
        "kappa_median": _safe_median(kappas),
        "n_steps_diss_dist": n_steps_diss,

        # Pass-through summary if present.
        "summary_file_present": os.path.exists(summary_path),
    }


def render_text(m: Dict, training_baseline: Optional[Dict[str, float]] = None) -> str:
    lines: List[str] = []
    sr = m["success_rate"]

    lines.append("=" * 72)
    lines.append(f"EVAL METRICS  --  {m['output_dir']}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("HEADLINE")
    lines.append("-" * 72)
    lines.append(
        f"  Episodes attempted   : {m['n_total']}"
    )
    lines.append(
        f"  Dissociated          : {m['n_dissociated']}  "
        f"(success rate = {_fmt(100*sr if sr is not None else None, '.1f')}%)"
    )
    lines.append("  Outcome breakdown    :")
    for k, v in sorted(m["outcome_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"      {k:<32s} {v}")
    lines.append("")

    lines.append("ACTIVATION ENERGY  (dissociated episodes only)")
    lines.append("-" * 72)
    lines.append(
        f"  <Ea_eff>  = {_fmt(m['ea_eff_mean'])} +/- {_fmt(m['ea_eff_std'])} eV     "
        f"median = {_fmt(m['ea_eff_median'])} eV"
    )
    if m["ea_eff_pcts"]:
        pcts = m["ea_eff_pcts"]
        lines.append(
            f"  Ea_eff percentiles: "
            f"P10={_fmt(pcts.get(10))}  P25={_fmt(pcts.get(25))}  "
            f"P50={_fmt(pcts.get(50))}  P75={_fmt(pcts.get(75))}  "
            f"P90={_fmt(pcts.get(90))}  eV"
        )
    lines.append(f"  <R_Ea>    = {_fmt(m['R_Ea_mean'])}     median = {_fmt(m['R_Ea_median'])}")
    lines.append(f"  <R_tot>   = {_fmt(m['R_tot_mean'])}")
    lines.append(f"  <steps>  to dissociation = {_fmt(m['steps_to_diss_mean'], '.1f')}     "
                 f"median = {_fmt(m['steps_to_diss_median'], '.1f')}")
    lines.append(f"  <NEB failures per dissociated episode> = "
                 f"{_fmt(m['neb_fails_per_diss_mean'], '.2f')}")
    lines.append("")

    if m["ea_eff_discrepancies"]:
        lines.append(f"  WARNING: {len(m['ea_eff_discrepancies'])} episodes have stored Ea_eff "
                     f"that differs from recomputed Ea_eff by more than 1 meV.")
        for ep, stored, recomp in m["ea_eff_discrepancies"][:5]:
            lines.append(f"      ep {ep}: stored={stored:.4f}  recomputed={recomp:.4f}")
        if len(m["ea_eff_discrepancies"]) > 5:
            lines.append(f"      ... ({len(m['ea_eff_discrepancies']) - 5} more)")
        lines.append("")
    else:
        lines.append("  OK Stored Ea_eff matches log-sum-exp recomputation from per-step Ea.")
        lines.append("")

    lines.append("STEP-LEVEL  (all valid steps, dissociated + non-dissociated episodes)")
    lines.append("-" * 72)
    lines.append(f"  Steps total          : {m['n_steps_total']}")
    lines.append(f"  Steps valid          : {m['n_valid_steps']}")
    lines.append(
        f"  Step Ea mean / median: {_fmt(m['step_ea_mean'])} / "
        f"{_fmt(m['step_ea_median'])} eV"
    )
    if m["step_ea_pcts"]:
        pcts = m["step_ea_pcts"]
        lines.append(
            f"  Step Ea percentiles  : "
            f"P10={_fmt(pcts.get(10))}  P25={_fmt(pcts.get(25))}  "
            f"P50={_fmt(pcts.get(50))}  P75={_fmt(pcts.get(75))}  "
            f"P90={_fmt(pcts.get(90))}  eV"
        )
    lines.append(
        f"  Step time mean/median: {_fmt(m['step_time_mean'], '.1f')}s / "
        f"{_fmt(m['step_time_median'], '.1f')}s    max = "
        f"{_fmt(m['step_time_max'], '.1f')}s"
    )
    lines.append(f"  Total step walltime  : {_fmt(m['total_step_walltime_h'], '.2f')} h")
    lines.append("")

    lines.append("ACTION SELECTION  (deterministic mode vs stochastic resample)")
    lines.append("-" * 72)
    lines.append(
        f"  MODE   : {m['mode_valid']:>5d} valid  /  {m['mode_invalid']:>5d} invalid     "
        f"(success rate = {100*m['mode_success_rate']:.1f}%)"
    )
    lines.append(
        f"  SAMPLE : {m['sample_valid']:>5d} valid  /  {m['sample_invalid']:>5d} invalid     "
        f"(success rate = {100*m['sample_success_rate']:.1f}%)"
    )
    n_steps = m["n_steps_total"] or 1
    pct_mode = 100 * (m["mode_valid"] + m["mode_invalid"]) / n_steps
    pct_samp = 100 * (m["sample_valid"] + m["sample_invalid"]) / n_steps
    lines.append(
        f"  Of all step attempts: {pct_mode:.1f}% used MODE on first try, "
        f"{pct_samp:.1f}% needed SAMPLE retry."
    )
    lines.append("")

    lines.append("POLICY DIAGNOSTICS")
    lines.append("-" * 72)
    lines.append(
        f"  vMF concentration (kappa)  : mean = {_fmt(m['kappa_mean'], '.2f')}     "
        f"median = {_fmt(m['kappa_median'], '.2f')}"
    )
    lines.append("")

    if training_baseline:
        lines.append("MANUSCRIPT COMPARISON")
        lines.append("-" * 72)
        lines.append(
            f"  {'metric':<32s} {'eval':>12s} {'training (cycle 3)':>22s}"
        )
        for key, label in [
            ("success_rate", "success rate"),
            ("ea_eff_mean", "<Ea_eff>  (eV)"),
            ("R_Ea_mean", "<R_Ea>"),
            ("steps_to_diss_mean", "<steps to dissociation>"),
        ]:
            tval = training_baseline.get(key)
            eval_val = m.get(key)
            if key == "success_rate":
                eval_str = (
                    f"{100*eval_val:.1f}%" if eval_val is not None else "--"
                )
                tstr = f"{100*tval:.1f}%" if tval is not None else "--"
            else:
                eval_str = _fmt(eval_val)
                tstr = _fmt(tval)
            lines.append(f"  {label:<32s} {eval_str:>12s} {tstr:>22s}")
        lines.append("")

    return "\n".join(lines)


def render_markdown(m: Dict, training_baseline: Optional[Dict[str, float]] = None) -> str:
    """Manuscript-friendly Markdown table."""
    sr = m["success_rate"]
    rows = [
        ("N episodes", str(m["n_total"])),
        ("N dissociated", str(m["n_dissociated"])),
        ("Success rate", f"{100*sr:.1f}%" if sr is not None else "--"),
        ("<Ea_eff> (eV)",
         f"{_fmt(m['ea_eff_mean'])} +/- {_fmt(m['ea_eff_std'])}"),
        ("Median Ea_eff (eV)", _fmt(m["ea_eff_median"])),
        ("<R_Ea>", _fmt(m["R_Ea_mean"])),
        ("<R_tot>", _fmt(m["R_tot_mean"])),
        ("<steps> to dissociation", _fmt(m["steps_to_diss_mean"], ".1f")),
        ("<NEB failures> per dissociated episode", _fmt(m["neb_fails_per_diss_mean"], ".2f")),
        ("MODE first-try success", f"{100*m['mode_success_rate']:.1f}%"),
        ("SAMPLE retry success", f"{100*m['sample_success_rate']:.1f}%"),
        ("Median step Ea (eV)", _fmt(m["step_ea_median"])),
        ("Median step time (s)", _fmt(m["step_time_median"], ".1f")),
        ("Total step walltime (h)", _fmt(m["total_step_walltime_h"], ".2f")),
    ]
    out = [f"# Eval metrics -- `{m['output_dir']}`", ""]
    out += ["| metric | value |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    out.append("")
    out.append("## Outcome breakdown")
    out.append("")
    out.append("| outcome | n |")
    out.append("|---|---|")
    for k, v in sorted(m["outcome_counts"].items(), key=lambda x: -x[1]):
        out.append(f"| {k} | {v} |")
    out.append("")

    if training_baseline:
        out.append("## Comparison to training cycle 3")
        out.append("")
        out.append("| metric | eval | training (cycle 3) |")
        out.append("|---|---|---|")
        labels = [
            ("success_rate", "success rate", True),
            ("ea_eff_mean", "<Ea_eff> (eV)", False),
            ("R_Ea_mean", "<R_Ea>", False),
            ("steps_to_diss_mean", "<steps>", False),
        ]
        for key, label, is_pct in labels:
            ev = m.get(key)
            tv = training_baseline.get(key)
            ev_s = (f"{100*ev:.1f}%" if is_pct else _fmt(ev)) if ev is not None else "--"
            tv_s = (f"{100*tv:.1f}%" if is_pct else _fmt(tv)) if tv is not None else "--"
            out.append(f"| {label} | {ev_s} | {tv_s} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True,
                   help="Eval output dir (containing per_episode.jsonl, per_step.jsonl).")
    p.add_argument("--markdown", action="store_true",
                   help="Emit Markdown table form instead of plain text.")
    p.add_argument("--json", action="store_true",
                   help="Emit raw JSON metrics dict (for piping into other tools).")
    p.add_argument("--with-training-baseline", action="store_true",
                   help="Include manuscript training-cycle-3 baseline in the comparison block.")
    args = p.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: not a directory: {args.output_dir}", file=sys.stderr)
        return 2

    metrics = analyze(args.output_dir)

    # Manuscript baseline numbers (training cycle 3, from your notes).
    training_baseline = None
    if args.with_training_baseline:
        training_baseline = {
            "success_rate": 0.67,
            "ea_eff_mean": 1.25,
            "R_Ea_mean": None,           # not in your earlier notes; leave blank
            "steps_to_diss_mean": None,
        }

    if args.json:
        # Strip non-serializable bits.
        printable = {k: v for k, v in metrics.items()
                     if k != "ea_eff_discrepancies"}
        printable["ea_eff_discrepancies"] = [
            {"episode": ep, "stored": stored, "recomputed": recomp}
            for ep, stored, recomp in metrics["ea_eff_discrepancies"]
        ]
        print(json.dumps(printable, indent=2))
    elif args.markdown:
        print(render_markdown(metrics, training_baseline))
    else:
        print(render_text(metrics, training_baseline))

    return 0


if __name__ == "__main__":
    sys.exit(main())
