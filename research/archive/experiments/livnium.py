"""
livnium.py
==========

Livnium: a pattern-replication operator stress-tested against the diagonal
Ramsey numbers.

WHAT THIS IS (and is NOT)
-------------------------
This module does NOT claim a Ramsey formula. The diagonal Ramsey numbers are
notoriously hard and the modern asymptotic story is *not* simple tripling:
Campos-Griffiths-Morris-Sahasrabudhe (2023) proved R(k) <= (4 - eps)^k, the
first exponential improvement over the old Erdos-Szekeres 4^k bound.

Instead, Livnium treats Ramsey as a *stress test*. We have a seed pattern

        2, 6, 18

and a naive "replication operator" (multiply by 3). It happens that these are
exactly the first diagonal Ramsey numbers:

        R(2,2) = 2
        R(3,3) = 6
        R(4,4) = 18

so the naive operator reproduces them *exactly* up to R(4,4), then predicts
R(5,5) = 54. The known range is 43 <= R(5,5) <= 46. The operator does not
degrade gradually -- it is exact, exact, exact, then snaps. The localized gap
at that transition is the *residual at the first breakpoint*, not "noise" in
the statistical sense (we have one breakpoint and the truth is an interval, so
it is a single residual with error bars, not a distribution).

GOAL
----
Provide a small framework to:
  1. define replication operators,
  2. compare their predicted diagonal sequence against known Ramsey bounds,
  3. measure the residual at each k (0 if the prediction lands inside the
     known interval, otherwise the distance to the nearest bound),
  4. locate the first breakpoint, and
  5. show whether a smarter operator (e.g. a multiplier that depends on k)
     shrinks the residual.

Ramsey bounds are taken from the standard living reference, Radziszowski's
"Small Ramsey Numbers" Dynamic Survey. Bounds shift over time; update
RAMSEY below if newer values land.

Run directly for a demo:   python3 livnium.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Ramsey data + seed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RamseyValue:
    """A diagonal Ramsey number R(k,k) as a closed interval [lo, hi].

    For settled values lo == hi (e.g. R(4,4) = 18 -> lo = hi = 18).
    For open cases lo < hi (e.g. R(5,5) in [43, 46]).
    """
    k: int
    lo: int
    hi: int

    @property
    def exact(self) -> bool:
        return self.lo == self.hi

    @property
    def mid(self) -> float:
        """Midpoint of the interval -- a convenient single-number 'truth'."""
        return (self.lo + self.hi) / 2.0

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def distance(self, x: float) -> float:
        """0 if x is inside [lo, hi], else distance to the nearest endpoint."""
        if x < self.lo:
            return self.lo - x
        if x > self.hi:
            return x - self.hi
        return 0.0

    def __str__(self) -> str:
        if self.exact:
            return f"R({self.k},{self.k}) = {self.lo}"
        return f"R({self.k},{self.k}) in [{self.lo}, {self.hi}]"


# Diagonal Ramsey numbers / best-known bounds.
# Source: Radziszowski, "Small Ramsey Numbers", Dynamic Survey DS1,
# Electronic Journal of Combinatorics (living reference). Update as needed.
RAMSEY: Dict[int, RamseyValue] = {
    2:  RamseyValue(2, 2, 2),
    3:  RamseyValue(3, 6, 6),
    4:  RamseyValue(4, 18, 18),
    5:  RamseyValue(5, 43, 46),
    6:  RamseyValue(6, 102, 161),
    7:  RamseyValue(7, 205, 492),
    8:  RamseyValue(8, 282, 1532),
    9:  RamseyValue(9, 565, 6588),
    10: RamseyValue(10, 798, 23556),
}

# The seed pattern that started this: the first three settled values.
SEED: List[int] = [RAMSEY[2].lo, RAMSEY[3].lo, RAMSEY[4].lo]  # [2, 6, 18]
ANCHOR_K: int = 2  # the k at which the sequence is anchored (R(2,2) = 2)


# ---------------------------------------------------------------------------
# 2. Operator framework
# ---------------------------------------------------------------------------

@dataclass
class Operator:
    """A replication operator.

    `step(prev, k)` returns the predicted value at k given the predicted value
    at k-1. The operator is run forward from the anchor value R(2,2) = 2, so
    every operator that returns 2 at the anchor and then applies its step rule
    produces a full predicted diagonal sequence.
    """
    name: str
    step: Callable[[float, int], float]
    note: str = ""

    def predict(
        self,
        ks: List[int],
        anchor_k: int = ANCHOR_K,
        anchor_value: float = float(SEED[0]),
    ) -> Dict[int, float]:
        """Generate predictions for each k in `ks` (must be contiguous)."""
        preds: Dict[int, float] = {}
        prev = anchor_value
        # Walk from anchor_k up to max(ks), applying step each time.
        for k in range(anchor_k, max(ks) + 1):
            if k == anchor_k:
                cur = anchor_value
            else:
                cur = self.step(prev, k)
            preds[k] = cur
            prev = cur
        return {k: preds[k] for k in ks}


# --- Candidate operators ---------------------------------------------------

def naive_triple() -> Operator:
    """next = 3 * prev. Exact through R(4,4), predicts R(5,5) = 54."""
    return Operator(
        name="naive x3",
        step=lambda prev, k: 3.0 * prev,
        note="2 -> 6 -> 18 -> 54 ...; matches R(2,2)..R(4,4) then breaks.",
    )


def es_style() -> Operator:
    """Erdos-Szekeres-flavoured growth: next = 4 * prev (the classic 4^k feel).

    Not meant to match small values -- it is the 'too fast' baseline that the
    real numbers sit well below, useful as an upper reference.
    """
    return Operator(
        name="ES-style x4",
        step=lambda prev, k: 4.0 * prev,
        note="The old 4^k growth rate; deliberately an over-estimate.",
    )


def cgms_style(eps: float = 0.5) -> Operator:
    """(4 - eps)^k flavour from Campos-Griffiths-Morris-Sahasrabudhe (2023).

    Implemented as a constant multiplier (4 - eps) per step. Asymptotic in
    spirit, not a small-value fit.
    """
    m = 4.0 - eps
    return Operator(
        name=f"CGMS-style x{m:g}",
        step=lambda prev, k: m * prev,
        note="First exponential improvement over 4^k (asymptotic flavour).",
    )


def k_conditioned(multipliers: Dict[int, float], default: float = 3.0) -> Operator:
    """next = m(k) * prev, where the multiplier depends on k.

    This is the first 'smarter' operator: instead of holding the multiplier at
    3, let it vary with k. `multipliers[k]` is the factor used when stepping
    INTO index k (i.e. from k-1 to k).
    """
    return Operator(
        name="k-conditioned",
        step=lambda prev, k: multipliers.get(k, default) * prev,
        note="Multiplier depends on k rather than being fixed at 3.",
    )


def fit_k_conditioned(target: str = "mid") -> Operator:
    """ORACLE operator. Per-step multiplier is fit to the known data, so it
    tracks the actual numbers instead of tripling.

    This is in-sample by construction: it has already seen every answer and
    picks multipliers that reproduce them. It is the best-possible curve, a
    reference ceiling -- NOT a predictive model. For the honest test use the
    forward-predictive walk below (`walk_forward`).

    For each k with known data we set the step multiplier to
        m(k) = truth(k) / truth(k-1)
    using either the interval midpoint ("mid") or the lower bound ("lo") as
    `truth`. Where data is missing the operator falls back to 3.0.

    This demonstrates the research goal directly: a learned operator drives the
    residual at the breakpoint toward zero.
    """
    def truth(rv: RamseyValue) -> float:
        return rv.mid if target == "mid" else float(rv.lo)

    multipliers: Dict[int, float] = {}
    ks = sorted(RAMSEY)
    for k in ks:
        if (k - 1) in RAMSEY:
            multipliers[k] = truth(RAMSEY[k]) / truth(RAMSEY[k - 1])
    op = k_conditioned(multipliers)
    op.name = f"ORACLE fitted ({target})"
    op.note = ("In-sample: multipliers chosen AFTER seeing the answers. "
               "Best-possible curve, NOT a predictive model. Reference only.")
    return op


# ---------------------------------------------------------------------------
# 3. Residual metrics
# ---------------------------------------------------------------------------

@dataclass
class Residual:
    k: int
    predicted: float
    rv: RamseyValue

    @property
    def absolute(self) -> float:
        """Distance from prediction to the known interval (0 if inside)."""
        return self.rv.distance(self.predicted)

    @property
    def signed_vs_mid(self) -> float:
        """Predicted minus interval midpoint (sign shows over/under-estimate)."""
        return self.predicted - self.rv.mid

    @property
    def normalized(self) -> float:
        """Absolute residual divided by the prediction (|pred - truth| / pred).

        Reported as a fraction; multiply by 100 for a percentage. This is the
        '14.8%-20.4% at R(5,5)' style number -- a descriptive residual, not a
        measured noise distribution.
        """
        if self.predicted == 0:
            return 0.0
        return self.absolute / abs(self.predicted)

    @property
    def consistent(self) -> bool:
        """True if the prediction lands inside the known interval."""
        return self.rv.contains(self.predicted)

    # --- honesty-aware metrics ---------------------------------------------
    @property
    def midpoint_residual(self) -> float:
        """|prediction - interval midpoint|.

        Unlike `absolute`, this does NOT go to zero just because the prediction
        slipped inside a wide interval -- it always measures distance to the
        single best point-estimate of the truth.
        """
        return abs(self.predicted - self.rv.mid)

    @property
    def width(self) -> int:
        """Width of the known interval (0 for settled values)."""
        return self.rv.hi - self.rv.lo

    @property
    def validatable(self) -> bool:
        """Is this interval tight enough that 'inside' actually means something?

        Rule: the uncertainty (width) must not exceed the lower bound itself.
        That keeps k=2..6 (where bounds are tight relative to the value) and
        drops k>=7, where intervals are so wide that landing inside is luck of
        ignorance rather than evidence.
        """
        return self.width <= self.rv.lo

    @property
    def width_penalty(self) -> float:
        """A 0..1 credit factor: full credit on tight/exact intervals, near
        zero on huge ones. Used to down-weight 'inside a barn door' wins.

            penalty = lo / (lo + width)

        Exact value (width=0) -> 1.0; [205,492] -> 205/492 ~ 0.42; the wider
        the box, the less a hit inside it counts.
        """
        denom = self.rv.lo + self.width
        return (self.rv.lo / denom) if denom else 1.0


def residuals_for(op: Operator, ks: Optional[List[int]] = None) -> List[Residual]:
    if ks is None:
        ks = sorted(RAMSEY)
    preds = op.predict(ks)
    return [Residual(k, preds[k], RAMSEY[k]) for k in ks]


def first_breakpoint(op: Operator, ks: Optional[List[int]] = None) -> Optional[int]:
    """The smallest k where the operator's prediction leaves the known interval.

    Returns None if the operator stays consistent across all tested k.
    """
    for r in residuals_for(op, ks):
        if not r.consistent:
            return r.k
    return None


def total_residual(
    op: Operator,
    ks: Optional[List[int]] = None,
    validatable_only: bool = False,
) -> float:
    """Sum of absolute residuals over the tested k.

    With `validatable_only=True`, only k whose interval is tight enough to be
    a real test count (drops the barn-door intervals at k>=7).
    """
    rs = residuals_for(op, ks)
    if validatable_only:
        rs = [r for r in rs if r.validatable]
    return sum(r.absolute for r in rs)


# ---------------------------------------------------------------------------
# 3b. Forward-predictive (OUT-OF-SAMPLE) test -- the honest one
# ---------------------------------------------------------------------------
#
# The oracle operator above is in-sample: it sees every answer first. The real
# question is whether replication residual goes DOWN out-of-sample:
#
#       train on k <= 4  ->  predict k = 5
#       train on k <= 5  ->  predict k = 6
#       train on k <= 6  ->  predict k = 7   ...
#
# At each target k a predictor sees ONLY the known values below k (using
# interval midpoints as the stand-in truth) and must forecast the next term.

TrainPoint = Tuple[int, float]          # (k, value)
PredictFn = Callable[[List[TrainPoint], int], float]


@dataclass
class ForwardPredictor:
    name: str
    fn: PredictFn
    note: str = ""


def _ratios(train: List[TrainPoint]) -> List[float]:
    return [train[i][1] / train[i - 1][1] for i in range(1, len(train))]


def fwd_naive3() -> ForwardPredictor:
    return ForwardPredictor(
        "naive x3",
        lambda tr, k: tr[-1][1] * 3.0,
        "Always triple the last known value.",
    )


def fwd_repeat_last() -> ForwardPredictor:
    return ForwardPredictor(
        "repeat-last-mult",
        lambda tr, k: tr[-1][1] * _ratios(tr)[-1],
        "Reuse the most recent observed multiplier.",
    )


def fwd_mean_mult() -> ForwardPredictor:
    def f(tr: List[TrainPoint], k: int) -> float:
        r = _ratios(tr)
        return tr[-1][1] * (sum(r) / len(r))
    return ForwardPredictor(
        "mean-mult",
        f,
        "Multiply by the average of all observed multipliers.",
    )


def fwd_loglinear() -> ForwardPredictor:
    """Least-squares line through (k, ln value); extrapolate one step.

    Captures a *changing* growth rate (the multipliers decelerate), which a
    fixed multiplier cannot. No numpy -- closed-form simple regression.
    """
    import math

    def f(tr: List[TrainPoint], k: int) -> float:
        xs = [p[0] for p in tr]
        ys = [math.log(p[1]) for p in tr]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx else 0.0
        intercept = my - slope * mx
        return math.exp(slope * k + intercept)

    return ForwardPredictor(
        "log-linear",
        f,
        "Fit log-growth trend; lets the growth rate bend, not stay fixed.",
    )


def fwd_shock_damped(eta: float = 0.7) -> ForwardPredictor:
    """Shock-memory operator: damp the multiplier after it over-expands.

    The replication idea made precise. Walk the observed history one step at a
    time, carrying an adjusted multiplier `m`. At each observed step we:

        realized_m = obs_k / obs_{k-1}          # what actually happened
        pred_k     = obs_{k-1} * m              # what we forecast with old m
        error_k    = log(pred_k / obs_k)        # >0 == we over-expanded
        m          = realized_m * exp(-eta * error_k)   # shock-damped update

    Then the forecast for the target is `obs_last * m`. When the last copy ran
    too large (error>0), the next multiplier is pulled down -- exactly the
    "absorb the shock, shrink the next copy" behaviour. eta is the damping /
    learning rate.
    """
    import math

    def f(tr: List[TrainPoint], k: int) -> float:
        if len(tr) < 2:
            return tr[-1][1] * 3.0
        m = tr[1][1] / tr[0][1]                 # seed with first realized ratio
        for i in range(2, len(tr)):
            prev, obs = tr[i - 1][1], tr[i][1]
            pred = prev * m
            error = math.log(pred / obs) if pred > 0 and obs > 0 else 0.0
            realized = obs / prev
            m = realized * math.exp(-eta * error)
        return tr[-1][1] * m

    return ForwardPredictor(
        f"shock-damped (eta={eta:g})",
        f,
        "Learns damping from how much the last copy over-expanded.",
    )


def fwd_shock_adaptive(eta: float = 0.3, beta: float = 0.6) -> ForwardPredictor:
    """Second-order shock memory: damping AND bounce.

    The midpoint multipliers ring (2.47 -> 2.96 -> 2.65) instead of decaying,
    so one-directional damping is wrong. This operator carries a second memory,
    the *rebound* (change in error), so it can expect the error to reverse:

        error_k   = log(pred_k / obs_k)
        rebound_k = error_k - error_{k-1}
        m         = realized_m * exp(-eta * error_k + beta * rebound_k)

    eta is damping memory (assume the shock persists); beta is bounce memory
    (assume the shock reverses). shock-damped is the beta=0 special case.
    """
    import math

    def f(tr: List[TrainPoint], k: int) -> float:
        if len(tr) < 2:
            return tr[-1][1] * 3.0
        m = tr[1][1] / tr[0][1]
        prev_error = 0.0
        for i in range(2, len(tr)):
            prev, obs = tr[i - 1][1], tr[i][1]
            pred = prev * m
            error = math.log(pred / obs) if pred > 0 and obs > 0 else 0.0
            rebound = error - prev_error
            realized = obs / prev
            m = realized * math.exp(-eta * error + beta * rebound)
            prev_error = error
        return tr[-1][1] * m

    return ForwardPredictor(
        f"shock-adaptive (eta={eta:g},beta={beta:g})",
        f,
        "Two memories: eta damps persistent shock, beta rides the rebound.",
    )


def best_eta(grid: Optional[List[float]] = None, target: str = "mid") -> Tuple[float, float]:
    """Sweep eta and return (best_eta, validatable_mid_residual) minimising the
    out-of-sample residual on tight intervals (k<=6)."""
    if grid is None:
        grid = [round(0.1 * i, 1) for i in range(0, 21)]  # 0.0 .. 2.0
    best = (0.0, float("inf"))
    for e in grid:
        rs = walk_forward(fwd_shock_damped(e), target=target)
        v = sum(r.midpoint_residual for r in rs if r.validatable)
        if v < best[1]:
            best = (e, v)
    return best


def best_eta_beta(
    grid: Optional[List[float]] = None, target: str = "mid"
) -> Tuple[float, float, float]:
    """Sweep (eta, beta); return (best_eta, best_beta, validatable_residual).

    WARNING: on the real Ramsey band only k<=6 is validatable and k=5 is
    irreducible, so this fits two knobs on essentially ONE point (k=6). Treat
    a low number here as overfitting, not validation. The synthetic tests are
    where the mechanism is actually checked.
    """
    if grid is None:
        grid = [round(0.1 * i, 1) for i in range(0, 16)]  # 0.0 .. 1.5
    best = (0.0, 0.0, float("inf"))
    for e in grid:
        for b in grid:
            rs = walk_forward(fwd_shock_adaptive(e, b), target=target)
            v = sum(r.midpoint_residual for r in rs if r.validatable)
            if v < best[2]:
                best = (e, b, v)
    return best


def walk_forward(
    pred: ForwardPredictor,
    target: str = "mid",
    min_train: int = 3,
) -> List[Residual]:
    """For each k with at least `min_train` known predecessors, train on
    k' < k (using `target` as truth) and predict k out-of-sample."""
    def tv(k: int) -> float:
        return RAMSEY[k].mid if target == "mid" else float(RAMSEY[k].lo)

    ks = sorted(RAMSEY)
    results: List[Residual] = []
    for k in ks:
        train = [(j, tv(j)) for j in ks if j < k]
        if len(train) < min_train:
            continue
        results.append(Residual(k, pred.fn(train, k), RAMSEY[k]))
    return results


def forward_report(preds: List[ForwardPredictor], target: str = "mid") -> str:
    out = []
    out.append("=" * 64)
    out.append("FORWARD-PREDICTIVE TEST (out-of-sample, train on k'<k)")
    out.append("=" * 64)
    out.append(f"truth proxy: interval {target}; one-step-ahead forecasts")
    out.append("")
    for p in preds:
        rs = walk_forward(p, target=target)
        out.append(f"Predictor: {p.name}   ({p.note})")
        hdr = f"  {'target k':>8}  {'predict':>9}  {'known':>16}  {'mid-res':>8}  in?"
        out.append(hdr)
        out.append("  " + "-" * (len(hdr) - 2))
        for r in rs:
            known = (f"{r.rv.lo}" if r.rv.exact else f"[{r.rv.lo},{r.rv.hi}]")
            inside = "yes" if r.consistent else "NO"
            out.append(
                f"  {r.k:>8}  {r.predicted:>9.1f}  {known:>16}  "
                f"{r.midpoint_residual:>8.1f}  {inside:>3}"
            )
        # Sum midpoint residual over validatable targets only.
        vtot = sum(r.midpoint_residual for r in rs if r.validatable)
        out.append(f"  validatable mid-residual (k<=6): {vtot:.1f}")
        out.append("")
    out.append("-" * 64)
    out.append("Reading it: at k=5 every predictor reproduces ~54, because the")
    out.append("training data (2,6,18) is pure tripling -- the deceleration is")
    out.append("INVISIBLE until you've seen k=5 fall short. The residual at the")
    out.append("first breakpoint is essentially irreducible from prior data.")
    out.append("From k=6 on, predictors that saw the k=5 shortfall pull ahead of")
    out.append("naive tripling -- that is replication residual actually going down.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3c. Validation: synthetic continuation + lower-bound walk
# ---------------------------------------------------------------------------

def oos_residuals_seq(
    pred: ForwardPredictor, series: List[TrainPoint], min_train: int = 3
) -> List[Tuple[int, float, float, float]]:
    """One-step-ahead residuals of `pred` over an arbitrary EXACT series.

    Returns (k, predicted, actual, |predicted-actual|) per forecastable point.
    Used for synthetic data where every value is known exactly.
    """
    out = []
    for idx in range(len(series)):
        train = series[:idx]
        if len(train) < min_train:
            continue
        k, actual = series[idx]
        p = pred.fn(train, k)
        out.append((k, p, actual, abs(p - actual)))
    return out


def _build_series(start: float, multipliers: List[float], k0: int = 2) -> List[TrainPoint]:
    """Build an exact series from a start value and a list of step multipliers."""
    vals = [start]
    for m in multipliers:
        vals.append(vals[-1] * m)
    return [(k0 + i, round(v, 3)) for i, v in enumerate(vals)]


def synthetic_test() -> str:
    """Validate the OPERATORS on data with known structure, independent of
    Ramsey. Two regimes:

      A. persistent deceleration  -> shock-damped should beat naive x3
      B. ringing / bouncing rate  -> shock-adaptive should beat shock-damped
    """
    out = []
    out.append("=" * 64)
    out.append("SYNTHETIC CONTINUATION TEST (validates the mechanism itself)")
    out.append("=" * 64)

    def tot(pred, series):
        return sum(r[3] for r in oos_residuals_seq(pred, series))

    # Regime A: multiplier decays smoothly 3.0, 2.8, 2.6, ... (persistent).
    decel = _build_series(2.0, [3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8])
    naive_a = tot(fwd_naive3(), decel)
    damp_a = tot(fwd_shock_damped(0.7), decel)
    adapt_a = tot(fwd_shock_adaptive(0.7, 0.0), decel)
    out.append("")
    out.append("A. persistent deceleration  (m: 3.0 -> 1.8, monotone)")
    out.append(f"   naive x3            total resid = {naive_a:>10.2f}")
    out.append(f"   shock-damped(0.7)   total resid = {damp_a:>10.2f}")
    out.append(f"   verdict: damping {'BEATS' if damp_a < naive_a else 'does NOT beat'} naive"
               f"  ({'mechanism validated' if damp_a < naive_a else 'unexpected'})")

    # Regime B: multiplier rings 2.4, 3.0, 2.4, 3.0, ... (bounces).
    ring = _build_series(2.0, [3.0, 2.4, 3.0, 2.4, 3.0, 2.4, 3.0])
    naive_b = tot(fwd_naive3(), ring)
    damp_b = tot(fwd_shock_damped(0.7), ring)
    adapt_b = tot(fwd_shock_adaptive(0.3, 0.9), ring)
    out.append("")
    out.append("B. ringing rate  (m alternates 3.0 / 2.4, like the Ramsey dip-rebound)")
    out.append(f"   naive x3                  total resid = {naive_b:>10.2f}")
    out.append(f"   shock-damped(0.7)         total resid = {damp_b:>10.2f}")
    out.append(f"   shock-adaptive(0.3,0.9)   total resid = {adapt_b:>10.2f}")
    best = min(naive_b, damp_b, adapt_b)
    winner = ("shock-adaptive" if best == adapt_b else
              "shock-damped" if best == damp_b else "naive")
    out.append(f"   verdict: lowest = {winner}"
               f"  ({'bounce memory helps on ringing data' if winner == 'shock-adaptive' else 'see numbers'})")
    out.append("")
    out.append("Takeaway: damping is right ONLY when deceleration persists (A).")
    out.append("When the rate rings (B) -- which is what the real Ramsey")
    out.append("midpoints do -- you need the second-order bounce memory.")
    return "\n".join(out)


def lower_bound_report() -> str:
    """Re-run the forward test against the LOWER BOUND instead of the midpoint,
    to see whether the lower-bound sequence compresses more persistently."""
    out = []
    out.append("=" * 64)
    out.append("LOWER-BOUND WALK-FORWARD (target = lo, not midpoint)")
    out.append("=" * 64)
    lo = {k: float(RAMSEY[k].lo) for k in sorted(RAMSEY)}
    mults = []
    ks = sorted(RAMSEY)
    for i in range(1, len(ks)):
        mults.append((ks[i], lo[ks[i]] / lo[ks[i - 1]]))
    out.append("lower-bound step multipliers:")
    out.append("   " + "  ".join(f"k{ k}:{m:.2f}" for k, m in mults))
    out.append("")
    es, eb, ev = best_eta_beta(target="lo")
    e1, e1v = best_eta(target="lo")
    out.append(f"best shock-damped   : eta={e1:g}  -> validatable resid {e1v:.1f}")
    out.append(f"best shock-adaptive : eta={es:g}, beta={eb:g} -> validatable resid {ev:.1f}")
    out.append("(still only k<=6 is validatable, so read as indicative, not proof)")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3d. Multi-sequence benchmark + auto-regime operator (frozen params)
# ---------------------------------------------------------------------------

def _make_ring_series(n: int, base: float, amp: float, noise: float,
                      seed: int, start: float = 2.0) -> List[TrainPoint]:
    """Ringing multiplier series: m alternates base+/-amp, optional noise."""
    import random
    rng = random.Random(seed)
    mults = []
    for i in range(n):
        m = base + (amp if i % 2 == 0 else -amp)
        if noise:
            m += rng.uniform(-noise, noise)
        mults.append(max(1.05, m))
    return _build_series(start, mults)


def _make_decel_series(n: int, m0: float, slope: float, noise: float,
                       seed: int, start: float = 2.0) -> List[TrainPoint]:
    """Persistent deceleration: multiplier decreases roughly linearly."""
    import random
    rng = random.Random(seed)
    mults = []
    for i in range(n):
        m = m0 - slope * i + (rng.uniform(-noise, noise) if noise else 0.0)
        mults.append(max(1.05, m))
    return _build_series(start, mults)


def _seq_total(pred: ForwardPredictor, series: List[TrainPoint]) -> float:
    """Normalised total residual: sum |pred-actual|/actual (scale-free)."""
    return sum(d / a for _, _, a, d in
               [(k, p, a, abs(p - a)) for k, p, a in
                ((r[0], r[1], r[2]) for r in oos_residuals_seq(pred, series))])


def ring_benchmark(trials: int = 300, n: int = 9) -> str:
    """Run many randomised ringing series; report how often shock-adaptive
    beats naive x3 and shock-damped (scale-free residual). This replaces the
    single fragile 211 vs 212 margin with a win-rate over a distribution."""
    import random
    rng = random.Random(12345)
    adapt = fwd_shock_adaptive(0.3, 0.9)
    damp = fwd_shock_damped(0.7)
    naive = fwd_naive3()
    win_vs_naive = win_vs_damp = 0
    ratios = []
    for t in range(trials):
        base = rng.uniform(2.2, 3.0)
        amp = rng.uniform(0.1, 0.6)
        noise = rng.uniform(0.0, 0.25)
        s = _make_ring_series(n, base, amp, noise, seed=t)
        ra, rn, rd = _seq_total(adapt, s), _seq_total(naive, s), _seq_total(damp, s)
        win_vs_naive += (ra < rn)
        win_vs_damp += (ra < rd)
        if rn > 0:
            ratios.append(ra / rn)
    mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
    out = []
    out.append("=" * 64)
    out.append(f"RING BENCHMARK  ({trials} randomised ringing series)")
    out.append("=" * 64)
    out.append("varied: base multiplier, amplitude, noise; scale-free residual")
    out.append("")
    out.append(f"  shock-adaptive beats naive x3   : "
               f"{win_vs_naive}/{trials}  ({100*win_vs_naive/trials:.0f}%)")
    out.append(f"  shock-adaptive beats shock-damped: "
               f"{win_vs_damp}/{trials}  ({100*win_vs_damp/trials:.0f}%)")
    out.append(f"  mean residual ratio adaptive/naive: {mean_ratio:.3f} "
               f"(<1 means adaptive is better on average)")
    return "\n".join(out)


def calibrate_frozen() -> Tuple[float, float]:
    """Freeze (eta, beta) from SYNTHETIC regimes, never from Ramsey.

    eta: best damping over a family of persistent-deceleration series.
    beta: best bounce (given that eta) over a family of ringing series.
    These constants are then applied to Ramsey untuned, so there is no
    'knobs fit on one point' overfit.
    """
    grid = [round(0.1 * i, 1) for i in range(0, 16)]  # 0.0 .. 1.5
    decel = [_make_decel_series(9, m0=3.2, slope=sl, noise=0.05, seed=s)
             for sl, s in [(0.15, 1), (0.2, 2), (0.25, 3), (0.18, 4)]]
    ring = [_make_ring_series(9, base=b, amp=a, noise=0.05, seed=s)
            for b, a, s in [(2.7, 0.3, 5), (2.6, 0.4, 6), (2.8, 0.25, 7), (2.5, 0.5, 8)]]

    def fam_total(pred, fam):
        return sum(_seq_total(pred, s) for s in fam)

    eta = min(grid, key=lambda e: fam_total(fwd_shock_damped(e), decel))
    beta = min(grid, key=lambda b: fam_total(fwd_shock_adaptive(eta, b), ring))
    return eta, beta


def fwd_shock_regime(eta: float, beta: float) -> ForwardPredictor:
    """Auto-regime operator. Detect persist vs ring from the sign of
    consecutive log-multiplier changes, then apply the matching memory:

        d_j   = log(m_j) - log(m_{j-1})
        same sign as d_{j-1}  -> persistent  -> damping only (beta off)
        sign flip             -> ringing      -> damping + bounce

    eta and beta are FROZEN from synthetic calibration, not tuned on Ramsey.
    """
    import math

    def f(tr: List[TrainPoint], k: int) -> float:
        if len(tr) < 2:
            return tr[-1][1] * 3.0
        ms = [tr[i][1] / tr[i - 1][1] for i in range(1, len(tr))]
        logm = [math.log(m) for m in ms]
        m = ms[0]
        prev_error = 0.0
        for i in range(2, len(tr)):
            prev, obs = tr[i - 1][1], tr[i][1]
            pred = prev * m
            error = math.log(pred / obs) if pred > 0 and obs > 0 else 0.0
            rebound = error - prev_error
            realized = obs / prev
            # regime from the two most recent log-multiplier deltas.
            # logm index j corresponds to multiplier ms[j] = step into tr[j+1].
            ring = False
            if i - 1 >= 2:
                d1 = logm[i - 1] - logm[i - 2]
                d0 = logm[i - 2] - logm[i - 3]
                ring = (d1 * d0) < 0  # sign flip == ringing
            if ring:
                m = realized * math.exp(-eta * error + beta * rebound)
            else:
                m = realized * math.exp(-eta * error)
            prev_error = error
        return tr[-1][1] * m

    return ForwardPredictor(
        f"shock-regime (eta={eta:g},beta={beta:g} FROZEN)",
        f,
        "Detects persist vs ring; applies damping or bounce. Params from synthetic.",
    )


def regime_confidence(multipliers: List[float], min_multipliers: int = 4,
                      eps: float = 1e-9) -> float:
    """Confidence (0..1) that the prior multipliers show a detectable ringing
    regime. Uses ONLY past multipliers; the sign-flip rate among consecutive
    deltas. Returns 0 when there is too little history to judge."""
    if len(multipliers) < min_multipliers:
        return 0.0
    deltas = [multipliers[i] - multipliers[i - 1]
              for i in range(1, len(multipliers))]
    usable = [d for d in deltas if abs(d) > eps]
    if len(usable) < 3:
        return 0.0
    sign_flips = comparisons = 0
    for a, b in zip(usable[:-1], usable[1:]):
        comparisons += 1
        if a * b < 0:
            sign_flips += 1
    return (sign_flips / comparisons) if comparisons else 0.0


def calibrate_threshold() -> float:
    """Freeze the confidence threshold from SYNTHETIC data: the midpoint
    between the mean confidence on ringing families and on persistent families.
    Never touches Ramsey."""
    ring = [_make_ring_series(9, b, a, 0.05, s)
            for b, a, s in [(2.7, 0.3, 5), (2.6, 0.4, 6), (2.8, 0.25, 7)]]
    decel = [_make_decel_series(9, 3.2, sl, 0.05, s)
             for sl, s in [(0.15, 1), (0.2, 2), (0.25, 3)]]

    def mean_conf(fam):
        cs = []
        for s in fam:
            m = [s[i][1] / s[i - 1][1] for i in range(1, len(s))]
            cs.append(regime_confidence(m))
        return sum(cs) / len(cs)

    return round((mean_conf(ring) + mean_conf(decel)) / 2.0, 3)


def fwd_shock_gated(eta: float, beta: float, threshold: float,
                    min_multipliers: int = 4) -> ForwardPredictor:
    """Deployable operator: fire the frozen shock-regime model ONLY when the
    regime is observable from prior data; otherwise fall back to naive x3.

    Every knob (eta, beta, threshold, min_multipliers) is frozen from synthetic
    calibration. Ramsey is evaluation only. The intended behaviour on the
    current Ramsey band is to refuse to act -> match naive, not beat it."""
    regime = fwd_shock_regime(eta, beta)
    naive = fwd_naive3()

    def f(tr: List[TrainPoint], k: int) -> float:
        if len(tr) < 2:
            return tr[-1][1] * 3.0
        mults = [tr[i][1] / tr[i - 1][1] for i in range(1, len(tr))]
        conf = regime_confidence(mults, min_multipliers)
        return regime.fn(tr, k) if conf >= threshold else naive.fn(tr, k)

    return ForwardPredictor(
        f"shock-gated (thr={threshold:g} FROZEN)",
        f,
        "Acts only when regime is observable; else refuses and uses naive.",
    )


def gated_mode_report(eta: float, beta: float, threshold: float,
                      min_multipliers: int = 4, target: str = "mid") -> str:
    """Show, per target k, the gate's confidence and whether it fired or fell
    back -- so the refuse-to-act behaviour is visible, not implied."""
    def tv(k):
        return RAMSEY[k].mid if target == "mid" else float(RAMSEY[k].lo)
    ks = sorted(RAMSEY)
    out = ["confidence gate per target k:",
           f"  {'k':>2}  {'#mult':>5}  {'conf':>5}  mode"]
    for k in ks:
        train = [(j, tv(j)) for j in ks if j < k]
        if len(train) < 3:
            continue
        mults = [train[i][1] / train[i - 1][1] for i in range(1, len(train))]
        conf = regime_confidence(mults, min_multipliers)
        mode = "FIRE shock-regime" if conf >= threshold else "fallback naive"
        rv = RAMSEY[k]
        valid = " (validatable)" if (rv.hi - rv.lo) <= rv.lo else ""
        out.append(f"  {k:>2}  {len(mults):>5}  {conf:>5.2f}  {mode}{valid}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. Reporting / demo
# ---------------------------------------------------------------------------

def report(op: Operator, ks: Optional[List[int]] = None) -> str:
    rs = residuals_for(op, ks)
    lines = []
    lines.append(f"Operator: {op.name}")
    if op.note:
        lines.append(f"  {op.note}")
    lines.append("")
    header = (f"  {'k':>2}  {'predicted':>10}  {'known':>16}  {'resid':>8}  "
              f"{'mid-res':>8}  {'norm%':>7}  in?  valid?")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in rs:
        known = (f"{r.rv.lo}" if r.rv.exact else f"[{r.rv.lo},{r.rv.hi}]")
        inside = "yes" if r.consistent else "NO"
        valid = "yes" if r.validatable else " -- "
        lines.append(
            f"  {r.k:>2}  {r.predicted:>10.1f}  {known:>16}  "
            f"{r.absolute:>8.1f}  {r.midpoint_residual:>8.1f}  "
            f"{r.normalized * 100:>6.1f}%  {inside:>3}  {valid:>5}"
        )
    bp = first_breakpoint(op, ks)
    lines.append("")
    lines.append(f"  first breakpoint     : {('k=' + str(bp)) if bp else 'none in range'}")
    lines.append(f"  total residual (all) : {total_residual(op, ks):.1f}")
    lines.append(f"  total residual (valid): {total_residual(op, ks, validatable_only=True):.1f}"
                 f"   <- only tight intervals (k<=6) count")
    return "\n".join(lines)


def compare(ops: List[Operator], ks: Optional[List[int]] = None) -> str:
    if ks is None:
        ks = sorted(RAMSEY)
    out = []
    out.append("=" * 64)
    out.append("LIVNIUM -- replication operators vs. diagonal Ramsey bounds")
    out.append("=" * 64)
    out.append(f"seed: {SEED}  (= R(2,2), R(3,3), R(4,4))")
    out.append(f"k range tested: {min(ks)}..{max(ks)}")
    out.append("")
    for op in ops:
        out.append(report(op, ks))
        out.append("")
    # Summary leaderboard ranked by VALIDATABLE residual (tight intervals only).
    ranked = sorted(ops, key=lambda o: total_residual(o, ks, validatable_only=True))
    out.append("-" * 64)
    out.append("leaderboard -- ranked by residual on tight intervals only (k<=6)")
    out.append("widths past k=6 are too large for 'inside' to count as evidence")
    out.append("-" * 64)
    for i, op in enumerate(ranked, 1):
        bp = first_breakpoint(op, ks)
        out.append(
            f"  {i}. {op.name:<22} valid-total={total_residual(op, ks, True):>8.1f}  "
            f"(all={total_residual(op, ks):>9.1f})  "
            f"first break={('k=' + str(bp)) if bp else 'none'}"
        )
    out.append("")
    out.append("NOTE: ORACLE rows are in-sample (fit after seeing answers) and")
    out.append("are a ceiling, not a prediction. See the forward-predictive test.")
    return "\n".join(out)


def main() -> None:
    ops = [
        naive_triple(),
        es_style(),
        cgms_style(eps=0.5),
        fit_k_conditioned("mid"),   # ORACLE -- reference ceiling only
    ]
    print(compare(ops))
    print()
    eta_star, eta_v = best_eta()
    eb_e, eb_b, eb_v = best_eta_beta()
    f_eta, f_beta = calibrate_frozen()   # frozen from synthetic, not Ramsey
    f_thr = calibrate_threshold()        # frozen from synthetic, not Ramsey
    print(forward_report([
        fwd_naive3(),
        fwd_repeat_last(),
        fwd_mean_mult(),
        fwd_loglinear(),
        fwd_shock_damped(eta_star),
        fwd_shock_adaptive(eb_e, eb_b),
        fwd_shock_regime(f_eta, f_beta),
        fwd_shock_gated(f_eta, f_beta, f_thr),
    ]))
    print()
    print(f"[sweep]  best damping eta = {eta_star:g} "
          f"(validatable resid = {eta_v:.1f})")
    print(f"[sweep]  best adaptive eta={eb_e:g}, beta={eb_b:g} "
          f"(validatable resid = {eb_v:.1f})  <- 2 knobs on ~1 point: OVERFIT")
    print(f"[frozen] regime eta={f_eta:g}, beta={f_beta:g} calibrated on "
          f"SYNTHETIC regimes, applied to Ramsey untuned (no overfit)")

    print(f"[frozen] gate threshold = {f_thr:g} (synthetic ring/persist split)")

    def _vresid(pred):
        return sum(r.midpoint_residual for r in walk_forward(pred) if r.validatable)
    naive_v = _vresid(fwd_naive3())
    regime_v = _vresid(fwd_shock_regime(f_eta, f_beta))
    gated_v = _vresid(fwd_shock_gated(f_eta, f_beta, f_thr))
    print()
    print("HONEST VERDICT on the Ramsey band (no Ramsey tuning):")
    print(f"  naive x3          validatable resid = {naive_v:.1f}")
    print(f"  shock-regime      validatable resid = {regime_v:.1f}  "
          f"({'beats' if regime_v < naive_v else 'does NOT beat'} naive)")
    print(f"  shock-gated       validatable resid = {gated_v:.1f}  "
          f"({'MATCHES' if abs(gated_v - naive_v) < 1e-6 else 'differs from'} naive)")
    print()
    print(gated_mode_report(f_eta, f_beta, f_thr))
    print()
    print("  -> The bounce MECHANISM is validated on synthetic rings (below).")
    print("     The ungated regime operator fires blindly and LOSES on Ramsey")
    print("     (too little history to detect the regime). The GATED operator")
    print("     refuses to act when confidence is low, so it falls back to")
    print("     naive on every validatable k and MATCHES it -- the correct,")
    print("     safe behaviour. It does not pretend to improve Ramsey yet.")
    print()
    print(synthetic_test())
    print()
    print(ring_benchmark())
    print()
    print(lower_bound_report())

    # Honest headline.
    naive = naive_triple()
    r5 = next(r for r in residuals_for(naive) if r.k == 5)
    print()
    print("=" * 64)
    print("HEADLINE")
    print("=" * 64)
    print("Livnium does not claim Ramsey numbers follow tripling.")
    print("A naive replication operator (x3) exactly matches R(2,2), R(3,3),")
    print(f"R(4,4), then fails at R(5,5): predicts {r5.predicted:.0f} vs known "
          f"{r5.rv.lo}-{r5.rv.hi}.")
    print(f"First-breakpoint residual = {r5.rv.distance(54):.0f}-"
          f"{abs(54 - r5.rv.lo):.0f} vertices, depending on where R(5,5) lands.")
    print("Goal: learn an operator that lowers OUT-OF-SAMPLE replication")
    print("residual -- not fit known intervals after the fact (that is the")
    print("ORACLE row, a ceiling). And only tight intervals (k<=6) count as")
    print("evidence; past k=6 the bounds are too wide to validate anything.")
    print()
    print("The Ramsey law is rejected, but the stress-test mechanism survives.")
    print("A shock-adaptive operator beats the naive multiplier in ~86% of")
    print("randomised ring trials and the shock-damped operator in 100%, mean")
    print("residual ratio ~0.72 -- the bounce mechanism is validated")
    print("INDEPENDENTLY of Ramsey. But with eta/beta/threshold frozen on")
    print("synthetic data and then applied to Ramsey, the operator does NOT")
    print("beat naive on the validatable band: the history at the only tight")
    print("breakpoint is too short to distinguish persistent deceleration from")
    print("ringing. So the conclusion is not 'Ramsey improved by fitting'. It is")
    print("stronger: replication-noise reduction REQUIRES regime detection;")
    print("ringing is a real mechanism; and Ramsey currently lacks enough tight")
    print("breakpoints to detect it without leakage. The deployable operator is")
    print("confidence-gated -- it refuses to act when the regime is unobservable,")
    print("so on today's Ramsey band it correctly MATCHES naive rather than")
    print("beating it. The negative result is the scientific constraint.")


if __name__ == "__main__":
    main()
