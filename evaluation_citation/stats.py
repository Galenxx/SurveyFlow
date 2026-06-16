"""Statistical primitives for Benchmark v2 (design doc §5).

The research critique flagged v1's statistics as unrigorous: one topic per
field, no repeats, no variance/CI, and percentages like "81.8%" reported off
a denominator of 11 (false precision). v2 fixes this with a strict protocol:

  * COUNTS FIRST. Every proportion is stored as (numerator, denominator).
    Percentages and CIs are a *derived* presentation layer, never the source
    of truth. The paper never prints a bare "81.8%" — it prints "9/11" with a
    Wilson interval.
  * Wilson score interval for every proportion (stable at small n, unlike the
    normal approximation which can leave [0,1] or collapse to width 0).
  * Paired comparison across the 10 TOPICS (the agreed pairing unit, n=10):
    SurveyFlow vs each baseline, Wilcoxon signed-rank.
  * Paired bootstrap over topics for the difference CI, plus the
    rank-biserial correlation effect size (the agreed effect-size measure).

This module is intentionally dependency-light: numpy + stdlib only (scipy is
NOT installed in this environment, and self-implementing keeps the exact
formulas auditable and quotable in the paper). All functions are pure.

Conventions:
  * A "proportion" is a (k, n) pair: k successes out of n trials.
  * A "paired sample" is two equal-length lists x, y where x[i] and y[i] are
    the same topic measured under two systems. Topics with a missing value on
    either side are dropped pairwise (and the effective n is reported).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np


# =====================================================================
# 1. Wilson score interval for a binomial proportion
# =====================================================================


@dataclass
class Proportion:
    """A proportion stored counts-first, with a Wilson CI on demand."""

    k: int                      # numerator (successes)
    n: int                      # denominator (trials)

    @property
    def point(self) -> Optional[float]:
        return (self.k / self.n) if self.n else None

    def wilson_ci(self, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
        return wilson_interval(self.k, self.n, z=z)

    def to_dict(self, z: float = 1.96) -> dict:
        lo, hi = self.wilson_ci(z=z)
        return {
            "k": self.k,
            "n": self.n,
            "point": self.point,
            "ci_low": lo,
            "ci_high": hi,
        }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    """Wilson score confidence interval for a binomial proportion k/n.

    Returns (low, high). For n == 0 returns (None, None). The Wilson interval
    is preferred over the normal (Wald) approximation because it stays inside
    [0, 1] and has good coverage even when k is 0, n, or n is small — exactly
    the regime here (n ~ 10 topics, k often near the extremes).

    Formula (z = 1.96 for 95%):
        center = (k + z^2/2) / (n + z^2)
        half   = z/(n+z^2) * sqrt( k(n-k)/n + z^2/4 )
    """
    if n <= 0:
        return (None, None)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


# =====================================================================
# 2. Wilcoxon signed-rank test (paired, two-sided)
# =====================================================================


@dataclass
class WilcoxonResult:
    statistic: float            # W = min(W+, W-) (after zero handling)
    p_value: float
    n_effective: int            # pairs actually used (after dropping/keeping zeros)
    n_zeros: int                # number of zero-difference pairs encountered
    method: str                 # "exact" | "normal-approx"
    zero_method: str            # how zero differences were treated

    def to_dict(self) -> dict:
        return asdict(self)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties get the mean of the rank span."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _wilcoxon_exact_pvalue(w_plus: float, n: int) -> Optional[float]:
    """Exact two-sided p-value for the Wilcoxon signed-rank null.

    Enumerates the 2^n sign assignments of ranks 1..n (no ties) to get the
    exact distribution of W+. Only feasible for small n; we cap at n<=20.
    Returns None when n is too large or ranks were tied (caller falls back to
    the normal approximation).
    """
    if n == 0 or n > 20:
        return None
    # distribution of W+ = sum of a subset of {1..n}
    max_sum = n * (n + 1) // 2
    # counts[s] = number of sign assignments giving W+ = s
    counts = np.zeros(max_sum + 1, dtype=np.float64)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        # adding rank can either contribute (shift) or not
        counts[rank:] = counts[rank:] + counts[:-rank] if rank <= max_sum else counts[rank:]
    total = 2.0 ** n
    # two-sided: p = P(W+ <= w) + P(W+ >= max_sum - w) but symmetric around mean
    w = int(round(w_plus))
    w = max(0, min(w, max_sum))
    lower = counts[: w + 1].sum()
    upper = counts[w:].sum()
    tail = min(lower, upper)
    p = 2.0 * tail / total
    return min(1.0, p)


def wilcoxon_signed_rank(
    x: Sequence[float],
    y: Sequence[float],
    zero_method: str = "pratt",
) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test on paired samples x, y.

    zero_method:
      - "pratt"  (default): zero differences are RANKED (kept in the ranking)
        but excluded from the test statistic. This is the honest choice when
        ties to zero are common, and it does NOT silently shrink n the way
        "wilcox" (drop zeros before ranking) does. We report n_effective and
        n_zeros so an n=10 comparison can never masquerade as something it is
        not.
      - "wilcox": drop zero differences entirely before ranking (classic).

    Uses the exact distribution when there are no ties and n_nonzero <= 20,
    otherwise the normal approximation with tie correction.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape:
        raise ValueError("x and y must have the same length")
    d = xa - ya
    n_zeros = int(np.sum(d == 0))

    if zero_method == "wilcox":
        d_nz = d[d != 0]
        rank_input = np.abs(d_nz)
        signs = np.sign(d_nz)
    elif zero_method == "pratt":
        # rank absolute differences INCLUDING zeros, then drop zeros from the
        # statistic (their signed contribution is zero anyway).
        rank_all = _rankdata(np.abs(d))
        mask = d != 0
        rank_input = None  # not used directly below
        signs = np.sign(d[mask])
        ranks = rank_all[mask]
    else:
        raise ValueError(f"unknown zero_method: {zero_method}")

    if zero_method == "wilcox":
        if len(rank_input) == 0:
            return WilcoxonResult(
                statistic=0.0, p_value=1.0, n_effective=0,
                n_zeros=n_zeros, method="degenerate", zero_method=zero_method,
            )
        ranks = _rankdata(rank_input)

    n_eff = int(len(ranks))
    if n_eff == 0:
        return WilcoxonResult(
            statistic=0.0, p_value=1.0, n_effective=0,
            n_zeros=n_zeros, method="degenerate", zero_method=zero_method,
        )

    w_plus = float(np.sum(ranks[signs > 0]))
    w_minus = float(np.sum(ranks[signs < 0]))
    w = min(w_plus, w_minus)

    # detect ties in the ranked magnitudes (affects exact feasibility)
    abs_for_ties = np.abs(d[d != 0]) if zero_method == "pratt" else rank_input
    _, tie_counts = np.unique(abs_for_ties, return_counts=True)
    has_ties = bool(np.any(tie_counts > 1))

    p_exact = None
    # The exact distribution is valid only when the ranks entering the test
    # are exactly 1..n_eff with no ties. That holds when there are no tied
    # magnitudes AND no zero differences are mixed into the ranking. Under
    # "wilcox" zeros are dropped before ranking, so no-ties is sufficient.
    # Under "pratt" zeros ARE ranked, so the kept ranks are 1..n_eff only when
    # there were no zeros at all; with zeros present the kept ranks skip the
    # zero block and the exact enumeration no longer matches.
    if not has_ties and n_zeros == 0:
        p_exact = _wilcoxon_exact_pvalue(w_plus, n_eff)

    if p_exact is not None:
        return WilcoxonResult(
            statistic=w, p_value=p_exact, n_effective=n_eff,
            n_zeros=n_zeros, method="exact", zero_method=zero_method,
        )

    # normal approximation with continuity correction + tie correction
    mean_w = n_eff * (n_eff + 1) / 4.0
    tie_term = float(np.sum(tie_counts ** 3 - tie_counts))
    var_w = (n_eff * (n_eff + 1) * (2 * n_eff + 1)) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return WilcoxonResult(
            statistic=w, p_value=1.0, n_effective=n_eff,
            n_zeros=n_zeros, method="degenerate", zero_method=zero_method,
        )
    z = (w_plus - mean_w)
    # continuity correction toward the mean
    if z > 0:
        z -= 0.5
    elif z < 0:
        z += 0.5
    z /= math.sqrt(var_w)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return WilcoxonResult(
        statistic=w, p_value=min(1.0, p), n_effective=n_eff,
        n_zeros=n_zeros, method="normal-approx", zero_method=zero_method,
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via erf (stdlib math.erf, no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# =====================================================================
# 3. Rank-biserial effect size (paired)
# =====================================================================


def rank_biserial(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Matched-pairs rank-biserial correlation as the effect size.

    r_rb = (W+ - W-) / sum(ranks), in [-1, +1]. Sign convention: positive
    means x tends to exceed y (SurveyFlow > baseline). |r| magnitude bands
    (Cohen-style, common convention): 0.1 small, 0.3 medium, 0.5 large.

    Computed with the Pratt convention (zeros ranked, excluded from the
    signed sums). Returns None when all pairs are tied.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    d = xa - ya
    mask = d != 0
    if not np.any(mask):
        return None
    ranks_all = _rankdata(np.abs(d))
    ranks = ranks_all[mask]
    signs = np.sign(d[mask])
    w_plus = float(np.sum(ranks[signs > 0]))
    w_minus = float(np.sum(ranks[signs < 0]))
    denom = w_plus + w_minus
    if denom == 0:
        return None
    return (w_plus - w_minus) / denom


# =====================================================================
# 4. Paired bootstrap for the difference of means (CI)
# =====================================================================


@dataclass
class BootstrapResult:
    diff_point: float           # observed mean(x) - mean(y)
    ci_low: float
    ci_high: float
    n_pairs: int
    n_resamples: int
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap_diff(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> Optional[BootstrapResult]:
    """Percentile bootstrap CI for the paired difference of means.

    Resamples TOPICS (the pairing unit) with replacement, recomputing
    mean(x_b) - mean(y_b) each time, then takes the percentile interval. The
    pairing is preserved: index i is resampled as a unit so x[i] and y[i]
    move together. Deterministic given `seed` (reproducibility, §8).

    Returns None when fewer than 2 complete pairs are available.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape:
        raise ValueError("x and y must have the same length")
    n = len(xa)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    diff_point = float(np.mean(xa) - np.mean(ya))
    idx = rng.integers(0, n, size=(n_resamples, n))
    xb = xa[idx]
    yb = ya[idx]
    diffs = xb.mean(axis=1) - yb.mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.percentile(diffs, 100 * alpha / 2.0))
    hi = float(np.percentile(diffs, 100 * (1.0 - alpha / 2.0)))
    return BootstrapResult(
        diff_point=diff_point,
        ci_low=lo,
        ci_high=hi,
        n_pairs=n,
        n_resamples=n_resamples,
        confidence=confidence,
    )


# =====================================================================
# 5. Convenience: full paired comparison bundle
# =====================================================================


@dataclass
class PairedComparison:
    """Everything needed to defend a 'SurveyFlow vs baseline' claim."""

    label: str                  # e.g. "SURVEYFLOW vs NAIVE_RAG / citation_recall"
    mean_x: Optional[float]
    mean_y: Optional[float]
    n_pairs: int
    wilcoxon: Optional[dict]
    rank_biserial: Optional[float]
    bootstrap: Optional[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def paired_comparison(
    label: str,
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> PairedComparison:
    """Run Wilcoxon + rank-biserial + paired bootstrap on one paired sample.

    Pairs where EITHER side is None/NaN are dropped pairwise first; the
    surviving n is reported as n_pairs throughout.
    """
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None
        and not (isinstance(a, float) and math.isnan(a))
        and not (isinstance(b, float) and math.isnan(b))
    ]
    if not pairs:
        return PairedComparison(
            label=label, mean_x=None, mean_y=None, n_pairs=0,
            wilcoxon=None, rank_biserial=None, bootstrap=None,
        )
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(pairs)
    wil = wilcoxon_signed_rank(xs, ys).to_dict() if n >= 1 else None
    rb = rank_biserial(xs, ys)
    boot = paired_bootstrap_diff(xs, ys, n_resamples=n_resamples, seed=seed)
    return PairedComparison(
        label=label,
        mean_x=float(np.mean(xs)),
        mean_y=float(np.mean(ys)),
        n_pairs=n,
        wilcoxon=wil,
        rank_biserial=rb,
        bootstrap=boot.to_dict() if boot else None,
    )
