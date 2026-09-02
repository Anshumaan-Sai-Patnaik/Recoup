"""B2 — Bandit (Arm Selection).

Decides *which* channel to retry next, re-consulted before every single attempt.
Learned statistics are pooled per context — (decline category, channel) — across
every customer and merchant that has ever produced that context, not kept
per-customer, per DESIGN.md's B2 mapping and the reasoning locked in in
notes/ARCHITECTURE.md Part B2. See notes/ARCHITECTURE.md Part B2 for the full role
description.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple, Optional

from recovery_agent import circuit_breaker
from recovery_agent.models import Customer, DeclineCategory, MandateChannel, TransactionState

DEFAULT_EPSILON = 0.1
"""Explore probability for `select_arm` — a demo-scale default (per DESIGN.md B2:
epsilon-greedy is a hand-rolled, explainable ~30-line algorithm, not tuned against
real traffic). Callers needing a different explore/exploit balance pass their own
`epsilon` to `select_arm` rather than relying on this constant."""


class BanditContext(NamedTuple):
    """The key the shared stats pool is organized by: "a soft decline on card" is a
    different learning context from "a hard decline on UPI," and so on.

    Deliberately keyed on the decline *category* (soft/hard), not the specific raw
    decline code — DESIGN.md's B2 mapping keys the stats dict on
    `(channel, decline_category)`, not `(channel, decline_code)`. Pooling at the
    category level (rather than one bucket per exact code) keeps each context's
    sample size large enough to actually learn something within a demo-scale batch;
    the Classifier (B1) has already thrown away the code-level detail that wouldn't
    generalize across codes anyway (e.g. "insufficient funds" and "issuer timeout"
    are different codes but both just "soft, worth retrying").
    """

    channel: MandateChannel
    decline_category: DeclineCategory


def context_key(channel: MandateChannel, decline_category: DeclineCategory) -> BanditContext:
    """Build the context key for one decision: which channel just declined, and
    what category (soft/hard) the Classifier (B1) assigned it. This is what the
    shared, cross-customer/cross-merchant stats pool (next task) is keyed on
    (ARCHITECTURE.md B2, operation 2) — e.g. every soft decline on card, from any
    customer of any merchant, contributes to and reads from the same
    `(card, soft)` bucket.
    """
    return BanditContext(channel=channel, decline_category=decline_category)


@dataclass
class ArmStats:
    """What's been learned so far about one arm (channel) within one context —
    just enough for an epsilon-greedy (or later UCB1) selection rule to work with.
    Starts at zero for a context/arm combination nobody has hit yet.
    """

    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        """The arm's observed success rate so far, or 0.0 if it's never been tried
        (an untried arm has no evidence either way — treating it as 0.0 rather than
        raising is what lets the selection rule fall back on its explore step to
        actually give it a first try, instead of crashing on unseen arms)."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts


class BanditStatsPool:
    """The shared, in-memory learning store the Bandit reads from and writes to.

    One pool instance is meant to live for the lifetime of a single batch run
    (or a test), and is genuinely shared — every customer, every merchant, every
    transaction that produces a given `BanditContext` reads and updates the exact
    same `ArmStats` object for that context/arm pair (ARCHITECTURE.md B2, operation
    3: pooled per context across *all* customers/merchants, never split out
    per-customer or per-merchant). DESIGN.md calls for "an in-memory dict keyed by
    context" — this class is a thin, purpose-built wrapper around exactly that dict,
    so callers get `get_stats`/`record_outcome` instead of reaching into nested
    dicts directly.
    """

    def __init__(self) -> None:
        self._stats: dict[BanditContext, dict[MandateChannel, ArmStats]] = {}

    def get_stats(self, context: BanditContext, channel: MandateChannel) -> ArmStats:
        """Read-only lookup of one arm's stats within one context. Returns a fresh
        zeroed `ArmStats` for a combination that's never been recorded, without
        writing anything into the pool — a mere read (e.g. the selection rule
        scanning every available arm) shouldn't grow the pool with empty entries
        for arms it only ever looked at and never actually chose.
        """
        return self._stats.get(context, {}).get(channel, ArmStats())

    def record_outcome(
        self, context: BanditContext, channel: MandateChannel, success: bool
    ) -> ArmStats:
        """Feed one attempt's real outcome back into the shared pool (the
        "learning" step, ARCHITECTURE.md B2 operation 5 — implemented fully in a
        later task, this is the storage half it will call). Unlike `get_stats`,
        this does create and persist an entry on first use, since an actual
        attempt happened and the pool must remember it for every future lookup.
        Returns the updated `ArmStats` for convenience (e.g. for the Audit Trail
        to cite the new running success rate).
        """
        arm_stats = self._stats.setdefault(context, {}).setdefault(channel, ArmStats())
        arm_stats.attempts += 1
        if success:
            arm_stats.successes += 1
        return arm_stats

    def reset(self) -> None:
        """Wipe the pool back to empty — mainly for test isolation, so one test's
        learned stats can never leak into the next."""
        self._stats.clear()


def _ordered_registered_channels(customer: Customer) -> list[MandateChannel]:
    """This customer's registered channels, in the order they were registered
    (so the primary channel — see `Customer.primary_channel()` — always sorts
    first), with duplicates dropped. `Customer.registered_channels()` returns a
    plain `set`, which has no stable order; the Bandit needs a deterministic
    ordering so arm selection is reproducible for a given seed.
    """
    seen: set[MandateChannel] = set()
    ordered: list[MandateChannel] = []
    for mandate in customer.mandates:
        if mandate.channel not in seen:
            seen.add(mandate.channel)
            ordered.append(mandate.channel)
    return ordered


def available_arms(
    customer: Customer,
    transaction: TransactionState,
    now: datetime,
    pending_decline: Optional[circuit_breaker.PendingDecline] = None,
) -> list[MandateChannel]:
    """The set of channels the Bandit is allowed to choose from.

    An arm is available only if it's both something this customer actually has
    registered (ARCHITECTURE.md B2 operation 1 — the agent can't spontaneously
    invent a mandate channel the customer never set up, per IDEA.md §5a) *and*
    not permanently closed by the Circuit Breaker (B4). The Bandit chooses only
    among whatever the Circuit Breaker still permits — it never overrules it
    (ARCHITECTURE.md B2 operation 6); a dead channel is never returned here,
    regardless of how favorably it's scored.

    "Permanently closed" (`circuit_breaker.is_channel_permanently_closed`) is the
    right test here rather than "closed right now" (`is_channel_open`), and the
    distinction matters: card's 24h soft-decline spacing rule is a *wait*, not a
    wall. Treating a channel that is merely serving out its spacing window as
    unavailable would make a card-only customer look like they had no options left
    and escalate them to a human without a single retry. Instead the arm stays
    selectable, and the Orchestrator (D1) simply doesn't fire the attempt until
    `circuit_breaker.earliest_next_attempt_at` permits it — so the rule is still
    obeyed exactly, just as a delay rather than a dead end. `now` is retained in
    the signature for that reason: callers reason about the same instant B4 does.

    `pending_decline` carries the original billing-event failure, which by this
    project's convention is not itself an attempt record — see
    `circuit_breaker.PendingDecline`. Without it, a channel that hard-declined on
    the original charge would still look available on the very first round.

    Returns an empty list if every registered channel is permanently closed —
    that's the Orchestrator's signal to route to Human Fallback (B5) instead of
    calling the Bandit at all (see `circuit_breaker.all_channels_permanently_closed`).
    """
    del now  # see docstring: availability is the permanent-closure question
    return [
        channel
        for channel in _ordered_registered_channels(customer)
        if not circuit_breaker.is_channel_permanently_closed(
            transaction, channel, pending_decline
        )
    ]


def select_arm(
    available_arms: list[MandateChannel],
    context: BanditContext,
    stats_pool: BanditStatsPool,
    rng: random.Random,
    epsilon: float = DEFAULT_EPSILON,
) -> MandateChannel:
    """Epsilon-greedy arm selection (ARCHITECTURE.md B2, operation 4).

    Chooses only among `available_arms` — the caller (the Orchestrator, D1) is
    responsible for having already filtered that list down via `available_arms()`
    above, so this function never has the chance to override the Circuit Breaker
    (B4, operation 6): there is no closed arm in scope for it to pick.

    With probability `epsilon`, explores: picks uniformly at random among
    `available_arms` via the supplied `rng`, so arm choice stays reproducible for
    a given seed (per DESIGN.md A2's seeded-reproducibility pattern, the same
    `random.Random` convention `simulator.py` already uses — never the global
    `random` module). Otherwise exploits: picks the arm with the highest observed
    `success_rate` in `context`, breaking ties by `available_arms` order (its
    first-registered-channel-first ordering from `_ordered_registered_channels`)
    rather than randomly, so a tie doesn't silently make an otherwise-seeded run
    non-reproducible.

    `available_arms` must be non-empty — an empty list means every registered
    channel is closed, which is the Orchestrator's signal to route to Human
    Fallback (B5) instead of calling the Bandit at all; it is not a case this
    function is meant to handle.
    """
    if not available_arms:
        raise ValueError("select_arm requires at least one available arm")

    if rng.random() < epsilon:
        return rng.choice(available_arms)

    best_arm = available_arms[0]
    best_rate = stats_pool.get_stats(context, best_arm).success_rate
    for arm in available_arms[1:]:
        rate = stats_pool.get_stats(context, arm).success_rate
        if rate > best_rate:
            best_arm = arm
            best_rate = rate
    return best_arm
