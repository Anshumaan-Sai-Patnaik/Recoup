"""B3 — Pacing (AIMD + Jitter).

Decides *when* the next attempt happens, once the Bandit (B2) has chosen an
arm. Two sub-responsibilities that sit together because jitter is applied
directly on top of AIMD's number, per DESIGN.md's B3 mapping. See
notes/ARCHITECTURE.md Part B3 for the full role description.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

DEGRADATION_FRACTION = 0.25
"""How much worse than our own norm the recent window has to be before AIMD
treats the system as degraded and backs off sharply.

**This replaces a fixed ~90% success-rate threshold, and the reason is worth
recording.** ARCHITECTURE.md B3 (operations 4–5) and IDEA.md §8b both name
"~90%" as the healthy line, borrowed directly from TCP congestion control.
That number is right in its home field, where virtually every packet arrives,
and wrong here: this system retries payments that have *already failed once*,
where an end-to-end batch success rate around 30% is a good day. A fixed 90%
line can therefore never be met, so additive increase would never fire, and
the dial would ratchet down to `MIN_AGGRESSIVENESS` on the first bad patch and
stay pinned there for the rest of the run — leaving a mechanism IDEA.md §12
calls load-bearing looking active while actually behaving as a constant ~48h
wait. This was caught by the Phase 8 end-to-end run, not by any unit test,
because it only shows up once real attempt volume flows through.

The fix keeps AIMD's shape and drops the borrowed constant: "healthy" is now
measured **against the system's own running baseline** rather than an absolute
number, so it self-calibrates to whatever success rate this population
actually produces. See `is_degraded` for the rule itself.

The specific fraction (25% worse than baseline) is our own choice, not a
borrowed figure, and should be described that way in any write-up. It is
deliberately a *relative* band so it stays meaningful whether the baseline
settles at 30% or 60%.
"""

MIN_SAMPLES_FOR_AIMD = 10
"""How many attempts must be on record before AIMD moves the dial at all.

Below this, recent and baseline are computed from nearly the same handful of
attempts, so comparing them says nothing — and acting on that noise would
swing the dial hardest exactly when the system knows least. Under this count
AIMD holds steady instead, which is the honest response to "not enough
evidence yet" and matches the same stance `RollingOutcomeWindow.success_rate`
and `bandit.ArmStats.success_rate` already take for an empty sample.
"""

ADDITIVE_INCREASE_STEP = 0.05
"""How much aggressiveness rises on a single healthy update — deliberately
small and fixed (the "additive" half of AIMD), so recovering trust in the
system is gradual even after a long healthy stretch."""

MULTIPLICATIVE_DECREASE_FACTOR = 0.5
"""What aggressiveness gets multiplied by on an unhealthy update — the
"multiplicative" half of AIMD. Halving (rather than a small fixed
subtraction) is the classic AIMD shape (same as TCP congestion control,
which is where this pattern is borrowed from per IDEA.md §8b): a bad
patch should cost much more than a good patch earns back, so the system
backs off fast and recovers slowly and cautiously."""

MIN_AGGRESSIVENESS = 0.05
"""Floor aggressiveness never drops below, even after repeated
multiplicative decreases — the system should always keep retrying
*eventually*, never fully freeze (that's the Circuit Breaker's/Human
Fallback's job, not Pacing's)."""

MAX_AGGRESSIVENESS = 1.0
"""Ceiling aggressiveness never rises above, so additive increase can't
compound past "as fast as the system is ever allowed to go.\""""

INITIAL_AGGRESSIVENESS = 0.5
"""Where a fresh batch run starts — a neutral midpoint, neither maximally
cautious nor maximally aggressive, since there's no evidence yet either
way about how healthy the system is."""

BASE_WAIT_MIN_HOURS = 1.0
"""The shortest base wait AIMD will ever hand back, at maximum
aggressiveness — deliberately matched to the low end of the Simulator's
own `recoverable_wait_hours_range` (`simulator.py`,
`SimulatorConfig.recoverable_wait_hours_range = (1.0, 48.0)`): 1 hour is
roughly the fastest a genuinely-recoverable transaction could plausibly
turn around, so it's meaningless for Pacing to ever retry sooner than
that even when it fully trusts the system."""

BASE_WAIT_MAX_HOURS = 48.0
"""The longest base wait AIMD will ever hand back, at minimum
aggressiveness — matched to the *high* end of that same
`recoverable_wait_hours_range`, for the same reason from the other
direction: when the system is unhealthy, back off all the way out to the
slowest plausible recovery window rather than inventing a separate,
disconnected timescale."""

JITTER_FRACTION = 0.2
"""Jitter is a bounded random offset within +/-20% of the base wait
(ARCHITECTURE.md B3 operation 7) — large enough to meaningfully break up
a thundering herd scheduled around the same base wait, small enough that
Jitter still reads as noise on top of AIMD's number rather than a second
source of pacing."""

JITTERED_WAIT_MIN_HOURS = 0.1
"""Floor on the final (base + jitter) wait, so a downward jitter roll at
the smallest base wait can never produce a zero or negative wait time."""


class RollingOutcomeWindow:
    """A system-wide rolling window of recent attempt outcomes.

    "System-wide" is the key distinction from the Circuit Breaker's (B4)
    per-transaction `AttemptRecord` lists (see notes/TRACKER.md's Phase 4
    notes): this window holds outcomes from *every* attempt across the whole
    simulated batch, regardless of which transaction, customer, or channel
    produced them, because AIMD (next task) reacts to overall system health,
    not any single transaction's history. DESIGN.md's B3 mapping calls for
    exactly a `collections.deque(maxlen=50)` for this — fixed size, O(1)
    push, nothing fancier needed — this class is a thin, purpose-built
    wrapper around exactly that, mirroring how `BanditStatsPool` wraps its
    own dict in bandit.py rather than handing callers a raw deque directly.
    """

    def __init__(self, maxlen: int = 50) -> None:
        self._outcomes: deque[bool] = deque(maxlen=maxlen)
        # Lifetime totals, kept alongside the rolling window rather than in a
        # second object: AIMD compares "how are we doing lately?" against "how
        # do we normally do?", and both halves of that question are answered by
        # the same stream of outcomes. Two counters are enough — there is no
        # need to retain every outcome ever seen just to hold an average.
        self._total_attempts = 0
        self._total_successes = 0

    def record(self, success: bool) -> None:
        """Append one attempt's outcome. Once the window is full (`maxlen`
        reached), each new outcome silently evicts the oldest one — the
        `deque(maxlen=...)` behavior that makes this a *rolling* window
        rather than an ever-growing log. The lifetime totals behind
        `baseline_success_rate` keep counting regardless of that eviction.
        """
        self._outcomes.append(success)
        self._total_attempts += 1
        self._total_successes += int(success)

    def __len__(self) -> int:
        return len(self._outcomes)

    @property
    def total_attempts(self) -> int:
        """Every attempt ever recorded in this batch run, including ones that
        have already rolled out of the window."""
        return self._total_attempts

    @property
    def baseline_success_rate(self) -> float:
        """The success rate across the *whole run so far* — "how do we normally
        do?" — as opposed to `success_rate`, which is "how are we doing lately?".

        This is what makes AIMD self-calibrating (see `DEGRADATION_FRACTION`):
        rather than measuring health against a borrowed constant that this
        domain can never reach, the system measures itself against its own
        track record, whatever success rate that turns out to be.

        Returns `1.0` on an empty run, matching `success_rate`'s neutral stance
        — though with no attempts recorded, `MIN_SAMPLES_FOR_AIMD` means AIMD
        will not be acting on this value anyway.
        """
        if self._total_attempts == 0:
            return 1.0
        return self._total_successes / self._total_attempts

    @property
    def success_rate(self) -> float:
        """The fraction of outcomes currently in the window that were
        successes — the "is the system healthy right now?" signal AIMD
        (next task) reacts to. Returns `1.0` on an empty window (no attempts
        recorded yet), the same "no evidence either way, don't punish it"
        stance `bandit.py`'s `ArmStats.success_rate` takes for an untried
        arm — except here `1.0` (not `0.0`) is the honest neutral default,
        since AIMD should hold steady rather than immediately slamming into
        multiplicative decrease before a single real attempt has happened.
        """
        if not self._outcomes:
            return 1.0
        return sum(self._outcomes) / len(self._outcomes)


SIGNAL_WARMING_UP = "warming_up"
SIGNAL_HEALTHY = "healthy"
SIGNAL_DEGRADED = "degraded"
"""The three things AIMD can conclude on a given round, named so the Audit
Trail (C1) can say which one drove a wait time without re-deriving it."""


def is_degraded(
    recent_success_rate: float, baseline_success_rate: float, sample_count: int
) -> bool:
    """The rule AIMD's multiplicative decrease fires on: is the system doing
    *meaningfully worse than it normally does*?

    Kept as a plain function over three numbers, rather than folded into
    `AimdState.update`, so the rule can be reasoned about and tested on its own
    without constructing a window or a dial.

    Two details matter, and both exist to protect AIMD's deliberately lopsided
    shape (a small `ADDITIVE_INCREASE_STEP` up, a hard
    `MULTIPLICATIVE_DECREASE_FACTOR` cut down):

    - The test is "worse than baseline **by a margin**" (`DEGRADATION_FRACTION`),
      not simply "below baseline." Recent performance sits below its own average
      roughly half the time by definition, so treating that as degraded would
      fire the sharp cut on half of all rounds and collapse the dial to its floor
      — the exact failure the fixed 90% threshold caused, arriving by a different
      route. Backing off has to be the *exception*, the way a dropped packet is
      in TCP, or AIMD stops being AIMD.
    - A baseline of zero with enough evidence counts as degraded. If nothing at
      all is working system-wide, "no deterioration from a norm of zero" is
      technically true and practically absurd; slow down.

    Below `MIN_SAMPLES_FOR_AIMD` this returns False (not degraded) — but callers
    should be using `pacing_signal` and holding the dial steady instead of
    reading that as a healthy signal.
    """
    if sample_count < MIN_SAMPLES_FOR_AIMD:
        return False
    if baseline_success_rate <= 0.0:
        return True
    return recent_success_rate < baseline_success_rate * (1.0 - DEGRADATION_FRACTION)


def pacing_signal(window: "RollingOutcomeWindow") -> str:
    """What the system's recent behavior says about its health right now —
    one of `SIGNAL_WARMING_UP`, `SIGNAL_HEALTHY`, or `SIGNAL_DEGRADED`.

    Separate from `AimdState.update` so a caller (the Orchestrator, and later
    the Audit Trail) can record *why* a wait came out the way it did without
    the act of asking changing the dial.
    """
    if window.total_attempts < MIN_SAMPLES_FOR_AIMD:
        return SIGNAL_WARMING_UP
    if is_degraded(
        window.success_rate, window.baseline_success_rate, window.total_attempts
    ):
        return SIGNAL_DEGRADED
    return SIGNAL_HEALTHY


@dataclass
class AimdState:
    """The single, system-wide "how eagerly should we retry right now?"
    dial (ARCHITECTURE.md B3, operations 3–5). One `AimdState` instance is
    meant to live for the lifetime of a whole batch run, the same way one
    `RollingOutcomeWindow` does — there is exactly one aggressiveness level
    for the entire system, not one per transaction or per channel, since
    Pacing's whole premise is reacting to overall system health rather than
    any single transaction's own history.

    `aggressiveness` is a plain float in `[MIN_AGGRESSIVENESS,
    MAX_AGGRESSIVENESS]`: higher means "the system currently trusts things
    are going well enough to retry sooner" (translated into an actual wait
    time by the next task); lower means "back off, retry later." It starts
    at `INITIAL_AGGRESSIVENESS` — a neutral guess — and only moves via
    `update()`.
    """

    aggressiveness: float = INITIAL_AGGRESSIVENESS

    def update(self, window: "RollingOutcomeWindow") -> float:
        """Apply one AIMD adjustment step from the system's current outcome
        history, and return the new aggressiveness level.

        Takes the whole `RollingOutcomeWindow` rather than a single success
        rate because the decision needs both halves of the comparison: how the
        system is doing *lately* (`success_rate`) versus how it does
        *normally* (`baseline_success_rate`). See `DEGRADATION_FRACTION` for
        why health is measured against the system's own baseline instead of the
        fixed ~90% line this originally used.

        Three outcomes, per `pacing_signal`:

        - **Warming up** — too few attempts on record to compare anything, so
          the dial holds steady. Acting on two or three data points would swing
          it hardest exactly when the system knows least.
        - **Healthy** — the normal case, and deliberately so: aggressiveness
          rises by a small fixed `ADDITIVE_INCREASE_STEP`, capped at
          `MAX_AGGRESSIVENESS`. A sustained good run nudges the system to retry
          a little sooner each time, gradually.
        - **Degraded** — recent results are meaningfully worse than this
          system's own norm (or nothing is working at all). Aggressiveness is
          cut to `MULTIPLICATIVE_DECREASE_FACTOR` of its current value, floored
          at `MIN_AGGRESSIVENESS`.

        The asymmetry is the whole point: one bad patch costs far more than one
        good round earns back, so the system is fast to back off and slow to
        recover — by design, not by accident. That only works while "healthy" is
        the common case, which is exactly what measuring against a self-
        calibrating baseline (rather than an unreachable constant) restores.
        """
        signal = pacing_signal(window)
        if signal == SIGNAL_WARMING_UP:
            return self.aggressiveness
        if signal == SIGNAL_HEALTHY:
            self.aggressiveness = min(
                self.aggressiveness + ADDITIVE_INCREASE_STEP, MAX_AGGRESSIVENESS
            )
        else:
            self.aggressiveness = max(
                self.aggressiveness * MULTIPLICATIVE_DECREASE_FACTOR, MIN_AGGRESSIVENESS
            )
        return self.aggressiveness

    def base_wait_hours(self) -> float:
        """Translate the current `aggressiveness` into a base wait time, in
        simulated hours, before the next attempt (ARCHITECTURE.md B3
        operation 6).

        Linear, inverse mapping across `[MIN_AGGRESSIVENESS,
        MAX_AGGRESSIVENESS]` onto `[BASE_WAIT_MIN_HOURS,
        BASE_WAIT_MAX_HOURS]`: higher aggressiveness (system trusted to be
        healthy) means a *shorter* wait, lower aggressiveness (system
        backing off) means a *longer* one. `MAX_AGGRESSIVENESS` maps
        exactly to `BASE_WAIT_MIN_HOURS` and `MIN_AGGRESSIVENESS` maps
        exactly to `BASE_WAIT_MAX_HOURS`; nothing here needs jitter yet —
        that's layered on top by `apply_jitter`.
        """
        span = MAX_AGGRESSIVENESS - MIN_AGGRESSIVENESS
        normalized = (self.aggressiveness - MIN_AGGRESSIVENESS) / span
        wait_span = BASE_WAIT_MAX_HOURS - BASE_WAIT_MIN_HOURS
        return BASE_WAIT_MAX_HOURS - normalized * wait_span


def apply_jitter(base_wait_hours: float, rng: random.Random) -> float:
    """Add a bounded random offset to a base wait (ARCHITECTURE.md B3
    operations 7-8), so transactions whose AIMD-computed base wait would
    otherwise land at the same simulated instant don't all fire in
    perfect synchrony (the thundering-herd condition the Simulator's
    "mass failure" mode, per ARCHITECTURE.md A2 operation 10, exists to
    exercise).

    The offset is drawn uniformly from +/- `JITTER_FRACTION` of the base
    wait, via a caller-supplied `random.Random` — never the global
    `random` module, matching the seeded-reproducibility convention
    already used throughout `simulator.py` and `bandit.py` — and the
    result is floored at `JITTERED_WAIT_MIN_HOURS` so a downward roll can
    never produce a zero or negative wait.
    """
    offset = rng.uniform(-JITTER_FRACTION, JITTER_FRACTION) * base_wait_hours
    return max(base_wait_hours + offset, JITTERED_WAIT_MIN_HOURS)
