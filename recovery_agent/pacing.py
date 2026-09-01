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

SUCCESS_RATE_THRESHOLD = 0.9
"""The "healthy system" line AIMD reacts to (ARCHITECTURE.md B3, operations
4–5: "~90%"). At or above this, the rolling success rate counts as healthy
and aggressiveness nudges up; below it, aggressiveness gets cut sharply."""

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

    def record(self, success: bool) -> None:
        """Append one attempt's outcome. Once the window is full (`maxlen`
        reached), each new outcome silently evicts the oldest one — the
        `deque(maxlen=...)` behavior that makes this a *rolling* window
        rather than an ever-growing log.
        """
        self._outcomes.append(success)

    def __len__(self) -> int:
        return len(self._outcomes)

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

    def update(self, success_rate: float) -> float:
        """Apply one AIMD adjustment step given the current rolling success
        rate (`RollingOutcomeWindow.success_rate`), and return the new
        aggressiveness level.

        At or above `SUCCESS_RATE_THRESHOLD`, this is "additive increase":
        aggressiveness rises by a small fixed `ADDITIVE_INCREASE_STEP`,
        capped at `MAX_AGGRESSIVENESS` — a sustained healthy run nudges the
        system to retry a little sooner each time, gradually.

        Below the threshold, this is "multiplicative decrease":
        aggressiveness is cut to `MULTIPLICATIVE_DECREASE_FACTOR` of its
        current value, floored at `MIN_AGGRESSIVENESS` — a single bad patch
        (or a whole cluster of failures, since both show up the same way in
        a rolling success rate that's dropped below 90%) costs far more
        than one healthy update earns back, which is what makes AIMD
        fast-to-back-off and slow-to-recover by design, not an accident.
        """
        if success_rate >= SUCCESS_RATE_THRESHOLD:
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
