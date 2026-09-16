"""Architectural invariant checks for the policy network.

`validate_architecture(policy, config=None)` introspects a constructed PolicyNetwork
and asserts the load-bearing structural invariants of the equivariant policy. Run it
right after the policy is built to catch silent architectural drift.

Two invariant classes:
  * Structural (introspected from `policy`): irreps, lmax/SH dim, multipole irreps,
    S2 grid resolution, RQS bins/layers, vMF axis head, learnable-beta bounds, frozen
    critic backbone.
  * Training-side literals in ppo/ppo_optimizer.py, not reachable from `policy`; reported,
    not asserted:
        - per-head loss weights  dir=1.0  mag=0.5  axis=0.05   (ppo_optimizer.py ~1376)
        - SH/direction grad clip sh_clip = 1.0                 (ppo_optimizer.py ~2341,2815)
        - per-head GAE lambda: see below.

Per-head GAE lambda: all three heads use the base gae_lambda (=0.95).
compute_per_head_advantages() computes
`dir_lambda = getattr(algorithm, 'direction_gae_lambda', gae_lambda)` but never uses it,
so the effective lambda is uniform 0.95 across dir/mag/axis. Surfaced here when a config
is supplied.
"""
from dataclasses import dataclass
from typing import Any, List, Optional


# Expected structural invariants.
EXPECTED_EMBEDDING_IRREPS = "128x0e+64x1e+32x2e"
EXPECTED_FRAME_ENCODER_OUTPUT_IRREPS = "16x0e+8x1e+4x2e"
EXPECTED_MULTIPOLE_IRREPS = "1x0e+1x1e+1x2e"
EXPECTED_LMAX = 4
EXPECTED_SH_DIM = (EXPECTED_LMAX + 1) ** 2  # 25
EXPECTED_S2_RES_BETA = 64
EXPECTED_S2_RES_ALPHA = 64
EXPECTED_RQS_NUM_BINS = 8
EXPECTED_RQS_NUM_LAYERS = 2
EXPECTED_LEARNABLE_BETA_INIT = 2.0
EXPECTED_LEARNABLE_BETA_MIN = 0.5
EXPECTED_LEARNABLE_BETA_MAX = 25.0
EXPECTED_ACTOR_UNFREEZE_MODE = "last_2_layers"

# Training-side literals preserved in ppo/ppo_optimizer.py (reported, not introspectable).
EXPECTED_LOSS_WEIGHTS = {"direction": 1.0, "magnitude": 0.5, "axis": 0.05}
EXPECTED_SH_GRAD_CLIP = 1.0
EFFECTIVE_GAE_LAMBDA_ALL_HEADS = 0.95  # effective uniform lambda for dir/mag/axis


@dataclass
class CheckResult:
    name: str
    expected: Any
    actual: Any
    ok: bool
    note: str = ""

    def __str__(self) -> str:
        mark = "OK  " if self.ok else "FAIL"
        line = f"[{mark}] {self.name}: expected={self.expected!r} actual={self.actual!r}"
        if self.note:
            line += f"  ({self.note})"
        return line


def _norm_irreps(value: Any) -> str:
    """Normalize an irreps string/object to a spaceless canonical string."""
    return str(value).replace(" ", "")


def validate_architecture(policy, config: Optional[Any] = None,
                          strict: bool = True) -> List[CheckResult]:
    """Validate the policy's architectural invariants.

    Args:
        policy: a constructed PolicyNetwork.
        config: optional PPOConfig; enables backbone / GAE-lambda cross-checks.
        strict: if True, raise AssertionError when any structural check fails.

    Returns:
        list of CheckResult. If strict and any check failed, raises before returning.
    """
    results: List[CheckResult] = []

    def check(name, expected, getter, note="", compare=None):
        try:
            actual = getter()
        except Exception as exc:  # missing attribute / wrong shape -> recorded as FAIL
            results.append(CheckResult(name, expected, f"<error: {exc}>", False, note))
            return
        if compare is not None:
            ok = compare(expected, actual)
        else:
            ok = (expected == actual)
        results.append(CheckResult(name, expected, actual, ok, note))

    irreps_eq = lambda e, a: _norm_irreps(e) == _norm_irreps(a)

    # 1. Per-O embedding irreps.
    check("embedding_irreps", EXPECTED_EMBEDDING_IRREPS,
          lambda: _norm_irreps(policy.irreps_atom), compare=irreps_eq)

    # 2. Frame-encoder output irreps.
    def _frame_out():
        fe = policy.frame_encoder
        if hasattr(fe, "get_output_irreps"):
            return _norm_irreps(fe.get_output_irreps())
        return _norm_irreps(getattr(fe, "output_irreps"))
    check("frame_encoder_output_irreps", EXPECTED_FRAME_ENCODER_OUTPUT_IRREPS,
          _frame_out, compare=irreps_eq)

    # 3. Direction-head lmax and SH vector length.
    check("sh_head.lmax", EXPECTED_LMAX, lambda: policy.sh_head.lmax)
    check("sh_head.sh_dim", EXPECTED_SH_DIM, lambda: policy.sh_head.sh_dim,
          note="(lmax+1)^2")

    # 4. Multipole injector input irreps (r_cut=9.0 lives on the multipole module,
    #    not the policy -- see environment multipole code; not checked here).
    check("multipole_injector.irreps_in", EXPECTED_MULTIPOLE_IRREPS,
          lambda: _norm_irreps(policy.multipole_injector.irreps_in), compare=irreps_eq)

    # 5. S2 / Gauss-Legendre grid resolution.
    check("sh_head.s2_block.res_beta", EXPECTED_S2_RES_BETA,
          lambda: policy.sh_head.s2_block.res_beta)
    check("sh_head.s2_block.res_alpha", EXPECTED_S2_RES_ALPHA,
          lambda: policy.sh_head.s2_block.res_alpha)

    # 6. RQS magnitude flow.
    check("magnitude_predictor.rqs_flow.num_bins", EXPECTED_RQS_NUM_BINS,
          lambda: policy.magnitude_predictor.rqs_flow.num_bins)
    check("magnitude_predictor.rqs_flow.num_layers", EXPECTED_RQS_NUM_LAYERS,
          lambda: policy.magnitude_predictor.rqs_flow.num_layers)

    # 7. vMF axis head present. Orthogonalization uses a Householder reflection
    #    (AxisSamplingManager._sample_vmf_householder), not Gram-Schmidt.
    check("axis_manager_present", True,
          lambda: policy.axis_manager is not None,
          note="vMF axis head uses Householder reflection (not Gram-Schmidt)")

    # 8. Learnable beta bounds (structural) and init (construction-time, soft).
    check("sh_head.learnable_beta_min", EXPECTED_LEARNABLE_BETA_MIN,
          lambda: policy.sh_head.learnable_beta_min)
    check("sh_head.learnable_beta_max", EXPECTED_LEARNABLE_BETA_MAX,
          lambda: policy.sh_head.learnable_beta_max)

    def _log_beta_init_ok(expected, actual):
        # Only meaningful on a freshly built policy; a trained/loaded policy will differ.
        return abs(float(actual) - float(expected)) < 1e-6
    check("sh_head.log_beta (init)", EXPECTED_LEARNABLE_BETA_INIT,
          lambda: float(policy.sh_head.log_beta.detach().cpu().item()),
          note="construction-time value; trained checkpoints differ",
          compare=_log_beta_init_ok)

    # 9. Critic embedder is non-learnable: CriticEmbedding composes [geom, sym_l0] with
    #    no nn.Parameter and no submodules; gradients reach the value head via value_mlp.
    #    is_frozen()/_frozen live on the critic_input_normalizer (unfrozen until post-warmup),
    #    not here. Structural invariant: the critic embedder has zero trainable parameters.
    #    Frozen critic *backbone* is the config-side backbone.critic_train_backbone==False check.
    def _critic_trainable_params():
        ce = policy.critic_embedder
        return sum(p.numel() for p in ce.parameters() if p.requires_grad)
    check("critic_embedder.non_learnable (0 trainable params)", 0,
          _critic_trainable_params,
          note="CriticEmbedding is pure math; gradients flow via value_mlp only")

    # Config-dependent cross-checks (only if a config is supplied).
    if config is not None:
        backbone = getattr(config, "backbone", None)
        if backbone is not None:
            check("backbone.unfreeze_mode (actor last-2-layers)",
                  EXPECTED_ACTOR_UNFREEZE_MODE,
                  lambda: backbone.unfreeze_mode)
            check("backbone.critic_train_backbone (frozen)", False,
                  lambda: backbone.critic_train_backbone)

        algo = getattr(config, "algorithm", config)
        has_dir_lambda = hasattr(algo, "direction_gae_lambda")
        results.append(CheckResult(
            "per_head_gae_lambda",
            f"all heads use base gae_lambda={EFFECTIVE_GAE_LAMBDA_ALL_HEADS}",
            (f"direction_gae_lambda attr present={has_dir_lambda}; "
             f"base gae_lambda={getattr(algo, 'gae_lambda', None)}"),
            ok=(not has_dir_lambda),
            note=("dir_lambda is computed via getattr but never used; "
                  "all heads advantaged with base gae_lambda (uniform 0.95)."),
        ))

    # Report-only training-side literals (not runtime-introspectable from policy).
    results.append(CheckResult(
        "per_head_loss_weights (source-preserved)", EXPECTED_LOSS_WEIGHTS,
        "see ppo/ppo_optimizer.py (local constants)", ok=True,
        note="source-preserved; not introspectable from policy"))
    results.append(CheckResult(
        "sh_grad_clip (source-preserved)", EXPECTED_SH_GRAD_CLIP,
        "see ppo/ppo_optimizer.py sh_clip", ok=True,
        note="source-preserved; not introspectable from policy"))

    failed = [r for r in results if not r.ok]
    if strict and failed:
        report = "\n".join(str(r) for r in results)
        raise AssertionError(
            "validate_architecture: "
            f"{len(failed)} invariant check(s) FAILED:\n{report}"
        )
    return results


def format_report(results: List[CheckResult]) -> str:
    """Render check results as a human-readable block."""
    return "\n".join(str(r) for r in results)


if __name__ == "__main__":
    # Self-describing usage: this module must be run with a constructed policy.
    print(__doc__)
