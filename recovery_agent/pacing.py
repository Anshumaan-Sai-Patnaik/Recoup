"""B3 — Pacing (AIMD + Jitter).

Decides *when* the next attempt happens, once the Bandit (B2) has chosen an
arm. Two sub-responsibilities that sit together because jitter is applied
directly on top of AIMD's number, per DESIGN.md's B3 mapping. See
notes/ARCHITECTURE.md Part B3 for the full role description.
"""

from __future__ import annotations

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
