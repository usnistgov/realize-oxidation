#!/usr/bin/env python3
"""
Estimate the lower-bound pathway time for the RL-discovered O2 oxidation
sequence using transition-state theory.

Two independent data sources are supported:
  * --per-episode + --per-step    (primary source, parsed from JSONL)
  * --stdout                      (optional; raw eval log cross-check)

For each dissociated episode the final accepted-step NEB barrier is the O2
dissociation barrier (prefactor nu_diss = 5e13 s^-1, from O-O stretch),
earlier accepted-step barriers are diffusion / hopping (nu_diff = 1e13 s^-1):

      k_i         = nu_i * exp(-E_i / (k_B T))
      tau_episode = sum_i 1 / k_i

Episodes accumulate in chronological order until cumulative dissociated O
atoms reaches --target-o (default 80). Equivalent MD-step count at 2 fs
is reported. If --stdout is provided, both sources run the same kinetic
calculation and a per-episode side-by-side comparison is printed; any
mismatch in barrier count or value beyond tolerance is a hard error.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

K_B_EV_PER_K  = 8.617333262e-5     # eV / K
NU_DISS       = 5.0e13             # s^-1, O-O stretch attempt frequency
NU_DIFF       = 1.0e13             # s^-1, conventional hopping prefactor
MD_TIMESTEP_S = 2.0e-15            # 2 fs

# Cross-check tolerances (JSONL stores barriers at higher precision than
# stdout's f-string; absolute eV tol of 1e-9 covers the formatting alone)
BARRIER_ABS_TOL = 1e-9             # eV
BARRIER_REL_TOL = 1e-9
CUM_TAU_REL_TOL = 1e-7             # cumulative tau is more sensitive


# ----------------------------------------------------------------------
# JSONL parsers
# ----------------------------------------------------------------------

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_valid_barriers_by_episode(steps):
    """Return {episode_idx: [E_i, ...]} of NEB-evaluated barriers per episode.

    Logger semantics (verified on eval_runs/ep550 logs, 2321 rows):
      * valid=False, ea=None     -> step rolled back (NEB failed / rejected).
      * valid=True,  ea=<float>  -> step accepted, NEB succeeded. Counted.
      * valid=True,  ea=None     -> step "happened" but terminated abnormally
                                    (done=true, d_oo=Infinity, reward_raw=-1.0).
                                    Not counted in per_episode.n_valid_neb.
    """
    raw = defaultdict(list)
    for s in steps:
        raw[s["episode"]].append(s)

    grouped = defaultdict(list)
    for ep, ep_steps in raw.items():
        ep_steps.sort(key=lambda x: x["step"])
        for s in ep_steps:
            if not s.get("valid", False):
                continue
            ea = s.get("ea", None)
            if ea is None:
                continue
            grouped[ep].append(float(ea))
    return grouped


# ----------------------------------------------------------------------
# Stdout parser
# ----------------------------------------------------------------------

# Header has the "(O incorporated" suffix so it can't collide with the
# similarly-prefixed summary line.
RE_EP_HEADER = re.compile(
    r"\[EVAL\]\s+Episode\s+(\d+)\s*\(O incorporated"
)
RE_EP_SUMMARY = re.compile(
    r"\[EVAL\]\s+Episode\s+(\d+)\s+summary:\s+"
    r"outcome=(\w+),\s+"
    r"steps=(\d+),\s+"
    r"Ea_eff=(\S+?),\s+"
    r"R_tot=(\S+?),\s+"
    r"R_Ea=(\S+)"
)
# Step line: leading whitespace, step <num>, [MODE] or [SAMPLE(try=K)],
# then the standard valid/d_OO/|s|/Ea/raw_R/done fields.
RE_STEP = re.compile(
    r"^\s*step\s+(\d+)\s+\[([^\]]+)\]\s+"
    r"valid=(True|False)\s+"
    r"d_OO=\S+\s+"
    r"\|s\|=\S+\s+"
    r"Ea=(\S+)\s+"
    r"raw_R=\S+\s+"
    r"done=(True|False)"
)


def parse_stdout(path):
    """Parse the eval stdout log into the same shape as the JSONL parsers.

    Logger quirk: some step lines are emitted twice with byte-identical
    content (verified on eval_runs/ep550 stdout: step 07 [MODE] appearing
    back-to-back). We dedupe consecutive identical step lines within an
    episode. Genuine retries differ in mode tag and/or numerics and are
    therefore preserved.

    Returns
    -------
    episodes : list[dict]
        One entry per episode that has a summary line, mirroring
        per_episode.jsonl: episode, outcome, dissociated, n_steps, ea_eff,
        R_tot, R_Ea, n_valid_neb (derived), cumulative_o_after (derived).
    barriers_by_ep : dict[int, list[float]]
        {episode: [E_i, ...]} in chronological order, only for valid=True
        steps with numeric Ea.
    """
    summaries = {}
    barriers  = defaultdict(list)
    current_ep = None
    last_step_line = None

    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = RE_EP_SUMMARY.search(line)
            if m:
                ep = int(m.group(1))
                ea_eff_s = m.group(4)
                ea_eff = None if ea_eff_s == "None" else float(ea_eff_s)
                summaries[ep] = dict(
                    episode=ep,
                    outcome=m.group(2),
                    n_steps=int(m.group(3)),
                    ea_eff=ea_eff,
                    R_tot=float(m.group(5)),
                    R_Ea=float(m.group(6)),
                    dissociated=(m.group(2) == "dissociated"),
                )
                continue

            m = RE_EP_HEADER.search(line)
            if m:
                current_ep = int(m.group(1))
                last_step_line = None
                continue

            m = RE_STEP.match(line)
            if m:
                if current_ep is None:
                    raise ValueError(
                        f"Step line found before any episode header: {line!r}"
                    )
                key = line.strip()
                if key == last_step_line:
                    continue
                last_step_line = key

                valid = (m.group(3) == "True")
                ea_s  = m.group(4)
                if not valid:
                    continue            # rolled-back step
                if ea_s == "None":
                    continue            # terminal-abnormal step
                barriers[current_ep].append(float(ea_s))

    # Build per-episode list with derived cumulative_o_after.
    episodes = []
    cum_O = 0
    for ep_idx in sorted(summaries):
        s = dict(summaries[ep_idx])
        if s["dissociated"]:
            cum_O += 2
        s["n_valid_neb"]        = len(barriers.get(ep_idx, []))
        s["cumulative_o_after"] = cum_O
        episodes.append(s)
    return episodes, barriers


# ----------------------------------------------------------------------
# Kinetic computation
# ----------------------------------------------------------------------

def episode_tau(barriers, kT):
    """tau_episode = sum_i 1/k_i. Last barrier uses NU_DISS, rest use NU_DIFF."""
    n = len(barriers)
    if n == 0:
        raise ValueError("Dissociated episode has zero valid NEB barriers.")
    tau = 0.0
    for i, E in enumerate(barriers):
        nu = NU_DISS if i == n - 1 else NU_DIFF
        tau += 1.0 / (nu * math.exp(-E / kT))
    return tau


def run_kinetic_calc(episodes, barriers_by_ep, kT, target_O, label):
    """Walk dissociated episodes, sum per-episode tau until cum_O >= target_O.

    Returns dict with rows = list of
        (ep_idx, n_NEB, E_diss, tau_ep, cum_O, cum_tau)
    plus aggregate cum_O, cum_tau, n_ep_used.
    """
    rows = []
    cum_O = 0
    cum_tau = 0.0
    for ep in episodes:
        if cum_O >= target_O:
            break
        if not ep.get("dissociated", False):
            continue
        ep_idx = ep["episode"]
        if ep_idx not in barriers_by_ep:
            raise ValueError(
                f"[{label}] Episode {ep_idx} dissociated but no step records "
                f"were parsed for it."
            )
        b = barriers_by_ep[ep_idx]
        if "n_valid_neb" in ep and len(b) != ep["n_valid_neb"]:
            raise ValueError(
                f"[{label}] Episode {ep_idx}: extracted {len(b)} valid NEB "
                f"barriers but n_valid_neb={ep['n_valid_neb']}."
            )
        tau_ep   = episode_tau(b, kT)
        cum_tau += tau_ep
        cum_O    = ep["cumulative_o_after"]
        rows.append((ep_idx, len(b), b[-1], tau_ep, cum_O, cum_tau))

    return dict(label=label, rows=rows,
                cum_O=cum_O, cum_tau=cum_tau, n_ep_used=len(rows))


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def _print_header(args, kT):
    print(f"# T = {args.temperature:g} K   (k_B T = {kT:.6f} eV)")
    print(f"# nu_diss = {NU_DISS:.2e} s^-1   nu_diff = {NU_DIFF:.2e} s^-1")
    print(f"# MD timestep = {MD_TIMESTEP_S*1e15:g} fs")
    print(f"# Target cumulative dissociated O atoms: {args.target_o}")
    print()


def _print_single_result(result, target_O):
    print(f"{'ep':>5} {'n_NEB':>6} {'E_diss(eV)':>11} "
          f"{'tau_ep(s)':>14} {'cum_O':>6} {'cum_tau(s)':>14}")
    print("-" * 64)
    for ep, n, E, tau_ep, cum_O, cum_tau in result["rows"]:
        print(f"{ep:>5d} {n:>6d} {E:>11.4f} "
              f"{tau_ep:>14.4e} {cum_O:>6d} {cum_tau:>14.4e}")
    print()
    print(f"Episodes consumed              : {result['n_ep_used']}")
    print(f"Cumulative O atoms dissociated : {result['cum_O']}")
    if result["cum_O"] < target_O:
        print(f"WARNING: target {target_O} not reached.")
    cum_tau = result["cum_tau"]
    print(f"Total pathway time tau         : {cum_tau:.6e} s")
    print(f"                                 = {cum_tau*1e9:.4f} ns")
    print(f"                                 = {cum_tau*1e6:.4f} us")
    print(f"                                 = {cum_tau*1e3:.4f} ms")
    print(f"Equivalent MD steps @ {MD_TIMESTEP_S*1e15:g} fs   : "
          f"{cum_tau / MD_TIMESTEP_S:.4e}")


def _print_comparison(j, s, j_b, s_b, target_O):
    """Side-by-side per-episode comparison + final consistency check."""
    j_rows = {row[0]: row for row in j["rows"]}
    s_rows = {row[0]: row for row in s["rows"]}
    common = sorted(set(j_rows) & set(s_rows))
    only_j = sorted(set(j_rows) - set(s_rows))
    only_s = sorted(set(s_rows) - set(j_rows))

    print(f"{'ep':>4} | {'n(J)':>4} {'n(S)':>4} | "
          f"{'E_diss(J)':>10} {'E_diss(S)':>10} | "
          f"{'tau_ep(J)':>12} {'tau_ep(S)':>12} | "
          f"{'rel_diff':>9}")
    print("-" * 80)
    for ep in common:
        jr = j_rows[ep]; sr = s_rows[ep]
        rel = abs(jr[3] - sr[3]) / max(abs(jr[3]), 1e-300)
        flag = "" if rel <= CUM_TAU_REL_TOL else "  *"
        print(f"{ep:>4d} | {jr[1]:>4d} {sr[1]:>4d} | "
              f"{jr[2]:>10.4f} {sr[2]:>10.4f} | "
              f"{jr[3]:>12.4e} {sr[3]:>12.4e} | "
              f"{rel:>9.2e}{flag}")

    if only_j:
        print(f"\nDissociated episodes in JSONL only  : {only_j}")
    if only_s:
        print(f"\nDissociated episodes in stdout only : {only_s}")
        # If stdout has an episode the JSONL doesn't, that's a real
        # cross-source inconsistency and we treat it as such.
        raise ValueError(
            f"Stdout reports dissociated episodes {only_s} that are not in "
            f"per_episode.jsonl."
        )

    # Strict per-barrier check on common episodes
    for ep in common:
        jb = j_b[ep]; sb = s_b[ep]
        if len(jb) != len(sb):
            raise ValueError(
                f"Episode {ep}: barrier count mismatch -- JSONL has {len(jb)} "
                f"valid NEBs, stdout has {len(sb)}."
            )
        for i, (a, b) in enumerate(zip(jb, sb)):
            if not math.isclose(a, b,
                                rel_tol=BARRIER_REL_TOL,
                                abs_tol=BARRIER_ABS_TOL):
                raise ValueError(
                    f"Episode {ep} barrier #{i}: JSONL={a!r} stdout={b!r}"
                )

    print()
    print(f"JSONL  cum_tau = {j['cum_tau']:.6e} s   "
          f"(cum_O = {j['cum_O']}, episodes = {j['n_ep_used']})")
    print(f"stdout cum_tau = {s['cum_tau']:.6e} s   "
          f"(cum_O = {s['cum_O']}, episodes = {s['n_ep_used']})")
    rel_total = abs(j["cum_tau"] - s["cum_tau"]) / max(abs(j["cum_tau"]), 1e-300)
    print(f"|delta tau| / tau = {rel_total:.2e}   "
          f"(tolerance = {CUM_TAU_REL_TOL:.0e})")
    if rel_total > CUM_TAU_REL_TOL:
        raise ValueError(
            f"JSONL and stdout disagree on cumulative tau: "
            f"|delta|/tau = {rel_total:.2e} > tol = {CUM_TAU_REL_TOL:.0e}."
        )

    cum_tau = j["cum_tau"]
    print()
    print(f"Total pathway time tau         : {cum_tau:.6e} s")
    print(f"                                 = {cum_tau*1e9:.4f} ns")
    print(f"                                 = {cum_tau*1e6:.4f} us")
    print(f"                                 = {cum_tau*1e3:.4f} ms")
    print(f"Equivalent MD steps @ {MD_TIMESTEP_S*1e15:g} fs   : "
          f"{cum_tau / MD_TIMESTEP_S:.4e}")
    if j["cum_O"] < target_O:
        print(f"WARNING: target {target_O} not reached "
              f"(reached {j['cum_O']}).")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--per-episode", default="per_episode.jsonl", type=Path)
    p.add_argument("--per-step",    default="per_step.jsonl",    type=Path)
    p.add_argument("--stdout",      default=None,                type=Path,
                   help="Optional eval stdout log; cross-checks JSONL.")
    p.add_argument("--temperature", default=1000.0, type=float, help="K")
    p.add_argument("--target-o",    default=80,     type=int,
                   help="Stop once cumulative_o_after reaches this value.")
    args = p.parse_args()

    for f in (args.per_episode, args.per_step):
        if not f.is_file():
            raise FileNotFoundError(f)
    if args.stdout is not None and not args.stdout.is_file():
        raise FileNotFoundError(args.stdout)

    kT = K_B_EV_PER_K * args.temperature
    _print_header(args, kT)

    # JSONL source (always)
    j_eps = load_jsonl(args.per_episode)
    j_eps.sort(key=lambda e: e["episode"])
    j_barriers = group_valid_barriers_by_episode(load_jsonl(args.per_step))
    j_result = run_kinetic_calc(j_eps, j_barriers, kT, args.target_o, "JSONL")

    if args.stdout is None:
        _print_single_result(j_result, args.target_o)
        return

    # Stdout source
    s_eps, s_barriers = parse_stdout(args.stdout)
    s_result = run_kinetic_calc(s_eps, s_barriers, kT, args.target_o, "STDOUT")

    _print_comparison(j_result, s_result, j_barriers, s_barriers, args.target_o)


if __name__ == "__main__":
    main()
