"""D1 — Orchestrator (Retry Loop Controller).

The only component that talks to all the others, and the point at which the Smart
Agent first becomes a runnable whole. It owns **no decision logic of its own** — it
sequences B1-B5 correctly, once per failed billing event and again for every retry
attempt within that transaction's lifetime, exactly per notes/ARCHITECTURE.md Part D1's
twelve numbered operations.

Every judgement call in here is a *sequencing* call, never a payments call: which
component to ask next, what to hand it, and when a transaction has finished. Whenever
this file looks like it is about to decide something substantive — is this decline
recoverable, may this channel be tried, which channel, how long to wait — it asks the
component that owns that question instead.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

from recovery_agent import bandit, circuit_breaker, pacing
from recovery_agent.bandit import (
    BanditContext,
    BanditStatsPool,
    available_arms,
    context_key,
    select_arm_with_mode,
)
from recovery_agent.classifier import UnrecognizedDeclineCodeError, classify
from recovery_agent.human_fallback import (
    HumanFallbackEvent,
    is_terminal,
    run_human_fallback,
)
from recovery_agent.models import (
    AttemptOutcome,
    AttemptRecord,
    BillingEvent,
    Customer,
    DeclineCategory,
    MandateChannel,
    Merchant,
    TransactionState,
    TransactionStatus,
)
from recovery_agent.pacing import AimdState, RollingOutcomeWindow, apply_jitter
from recovery_agent.simulator import SimulatedClock, Simulator

ARM_SAME_CHANNEL = "same_channel_same_route"
ARM_SWITCH_CHANNEL = "switch_channel"
"""The two arm labels this build can actually distinguish, recorded on every
`AttemptRecord.chosen_arm` for the Audit Trail (C1) and the Metrics engine (C3) to read.

IDEA.md §5a lists four arms; only these two are real here. Arm 2 (same channel,
different *route*) isn't modelled — the Simulator's Hidden Truth varies outcomes by
channel, not by route within a channel, so a route switch would be a distinction the
simulated world cannot honour (already recorded as a Phase 2 deviation in
notes/TRACKER.md). Arm 4 (human fallback) is deliberately not a bandit arm at all: B5
is reached only when the Circuit Breaker has closed everything else, never by the
Bandit choosing it (ARCHITECTURE.md B2/B5).
"""


@dataclass
class OrchestratorConfig:
    """Knobs for the loop's own sequencing, not for any component's decisions.

    Everything a component decides for itself (Visa caps, aggressiveness bounds,
    epsilon's default) lives with that component; only the two stopping rules the
    Orchestrator genuinely owns are here.
    """

    seed: int = 0
    """Seeds the one `random.Random` shared by arm selection and jitter, so a whole
    batch run reproduces exactly — the same seeded-reproducibility convention
    `simulator.py` follows (A2 operation 9), extended to the agent's own choices so
    that "same seed, same world" also means "same seed, same decisions"."""

    epsilon: float = 0.1
    """Explore probability handed to the Bandit's epsilon-greedy rule."""

    recovery_horizon_days: float = 30.0
    """How long after a billing event's scheduled time the agent keeps trying before
    marking the transaction `ABANDONED`.

    This is the Orchestrator's own stopping rule and the only thing that ever produces
    `ABANDONED`, which is what distinguishes it from `ESCALATED_TO_HUMAN`: escalation
    means *we ran out of options* (the Circuit Breaker closed every channel), abandonment
    means *we ran out of time* (the channels are still open, but this billing cycle is
    over and the next charge supersedes it). 30 days matches the default billing cycle
    in `SimulatorConfig.billing_cycle_days` for exactly that reason.
    """

    max_attempts_per_transaction: int = 25
    """A belt-and-braces cap on total attempts for one transaction, above the sum of
    every channel's own cap (15 card + 5 UPI + 5 netbanking = 25). It should never be
    the binding constraint — the Circuit Breaker's caps and the horizon above should
    always bite first — and exists only so that a future change to a channel rule can
    never turn this loop into an infinite one."""

    transaction_id_prefix: str = "txn"
    """Prefixes generated transaction ids. The Baseline Agent (C2, Phase 9) runs over
    the identical billing events and needs its transactions to be distinguishable from
    the Smart Agent's when the Metrics engine (C3) reads both."""


@dataclass
class SmartAgentRuntime:
    """The state that is deliberately **shared across the whole batch**, not per
    transaction.

    This is the difference between a batch of independent retries and an agent that
    learns: the Bandit's stats pool (B2 operation 3) pools evidence across every
    customer and merchant, and Pacing's rolling window plus AIMD dial (B3 operations
    1-5) describe the health of the *system*, not of any one transaction. Creating a
    fresh runtime per transaction would silently delete both mechanisms while leaving
    the code looking correct, so they are grouped here in one object with one lifetime:
    one batch run.
    """

    stats_pool: BanditStatsPool = field(default_factory=BanditStatsPool)
    outcome_window: RollingOutcomeWindow = field(default_factory=RollingOutcomeWindow)
    aimd: AimdState = field(default_factory=AimdState)
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    @classmethod
    def from_config(cls, config: OrchestratorConfig) -> "SmartAgentRuntime":
        return cls(rng=random.Random(config.seed))


ATTEMPT_DECISION_EVENT_TYPE = "attempt_decision"
"""The `event_type` tag every per-attempt record this component emits carries.

Tagged at the source, following the convention
`human_fallback.HUMAN_FALLBACK_EVENT_TYPE` set: the Audit Trail (C1) collects raw events
from several components and picks a renderer per type, and letting it guess from an
event's shape would be both slower and more fragile than each component simply saying
what it produced.
"""


@dataclass(frozen=True)
class AttemptDecision:
    """One round of the loop, recorded as facts.

    ARCHITECTURE.md D1 operation 10 asks for a decision event to be emitted for every
    choice made in steps 2-8. The Audit Trail (C1) that will render those into English
    isn't built until Phase 10, so this records the *facts* each round turned on and
    leaves the sentence-writing to C1 — the same facts-not-prose stance
    `human_fallback.HumanFallbackEvent` already takes, and for the same reason: a log
    line that asserts a "because" the system never actually computed is not an audit
    record.

    Frozen, again for the same reason: a decision record that can be edited after the
    fact isn't a record.
    """

    transaction_id: str
    attempt_number: int
    decided_at: datetime
    context_channel: MandateChannel
    context_category: DeclineCategory
    channel_status: tuple[tuple[MandateChannel, str], ...]
    available_arms: tuple[MandateChannel, ...]
    chosen_channel: MandateChannel
    chosen_arm: str
    rolling_success_rate: float
    baseline_success_rate: float
    health_signal: str
    aggressiveness: float
    base_wait_hours: float
    jittered_wait_hours: float
    actual_wait_hours: float
    spacing_rule_applied: bool
    attempted_at: datetime
    outcome: AttemptOutcome
    decline_code: Optional[str] = None
    decline_category: Optional[DeclineCategory] = None
    unrecognized_decline_code: bool = False
    selection_mode: str = bandit.SELECTION_EXPLOIT
    """Which half of epsilon-greedy chose `chosen_channel` — `"exploit"` (the best
    observed record in this context) or `"explore"` (a deliberate random draw).

    Recorded because C1 has to describe the choice in English and the two are different
    reasons. Not derivable after the fact: an exploration can land on the same arm an
    exploit would have picked, so only the Bandit itself knows which happened.
    """

    event_type: str = ATTEMPT_DECISION_EVENT_TYPE

    def to_dict(self) -> dict[str, Any]:
        """A flat, JSON/CSV-friendly row, so C1's `pandas` export (ARCHITECTURE.md C1
        operation 4) needs no per-event special-casing."""
        return {
            "event_type": self.event_type,
            "transaction_id": self.transaction_id,
            "attempt_number": self.attempt_number,
            "decided_at": self.decided_at.isoformat(),
            "context_channel": self.context_channel.value,
            "context_category": self.context_category.value,
            "channel_status": ",".join(f"{c.value}:{s}" for c, s in self.channel_status),
            "available_arms": ",".join(c.value for c in self.available_arms),
            "chosen_channel": self.chosen_channel.value,
            "chosen_arm": self.chosen_arm,
            "selection_mode": self.selection_mode,
            "rolling_success_rate": self.rolling_success_rate,
            "baseline_success_rate": self.baseline_success_rate,
            "health_signal": self.health_signal,
            "aggressiveness": self.aggressiveness,
            "base_wait_hours": self.base_wait_hours,
            "jittered_wait_hours": self.jittered_wait_hours,
            "actual_wait_hours": self.actual_wait_hours,
            "spacing_rule_applied": self.spacing_rule_applied,
            "attempted_at": self.attempted_at.isoformat(),
            "outcome": self.outcome.value,
            "decline_code": self.decline_code,
            "decline_category": (
                self.decline_category.value if self.decline_category else None
            ),
            "unrecognized_decline_code": self.unrecognized_decline_code,
        }


@dataclass
class TransactionResult:
    """Everything one transaction's journey produced: its final state, the per-attempt
    decision records, and the terminal human-fallback event if it ended that way
    (ARCHITECTURE.md D1, Outputs)."""

    transaction: TransactionState
    decisions: list[AttemptDecision] = field(default_factory=list)
    human_fallback_event: Optional[HumanFallbackEvent] = None

    origin_channel: Optional[MandateChannel] = None
    origin_category: Optional[DeclineCategory] = None
    origin_unrecognized: bool = False
    """The Classifier's (B1) call on the **original billing-event failure** — the very
    first diagnosis this journey turned on, recorded here because nothing else keeps it.

    The original charge is deliberately not an `AttemptRecord` (the Phase 2 convention),
    and while the opening diagnosis is normally visible as the first decision's
    `context_category`, a transaction whose only channel is hard-declined escalates on
    round 1 and emits **no decision at all** — 16% of a typical batch. Its opening call
    would then exist nowhere, and the Metrics engine (C3, operation 7) would have to
    re-run the Classifier to recover it. Re-running a decision to find out what it was
    measures the measurer's copy of the rule rather than the run, so the component that
    made the call records it instead. Same reasoning as
    `AttemptDecision.selection_mode`: only the component that decided can honestly say
    what it decided.

    `origin_unrecognized` is `True` when B1 refused to classify the opening code, in
    which case `origin_category` stays `None` rather than being filled with a guess.
    """

    terminal_reason: str = ""
    """A short machine-readable tag for *why* the loop stopped — `"recovered"`,
    `"all_channels_closed"`, `"recovery_horizon_exceeded"`, `"max_attempts_reached"`,
    or `"unrecognized_decline_code"`. Kept as a tag rather than a sentence so C1 renders
    the English and this file never asserts prose it would have to keep in sync."""


def _arm_label(chosen: MandateChannel, previous: MandateChannel) -> str:
    """Which of IDEA.md §5a's arms this choice corresponds to — see `ARM_SAME_CHANNEL`
    for why only two of the four are distinguishable here."""
    return ARM_SAME_CHANNEL if chosen == previous else ARM_SWITCH_CHANNEL


def _classify_decline(
    channel: MandateChannel, code: str
) -> tuple[Optional[DeclineCategory], bool]:
    """ARCHITECTURE.md D1 operation 2 — hand the decline code to the Classifier (B1).

    Returns `(category, unrecognized)`. An unrecognized (channel, code) pair comes back
    as `(None, True)` rather than a guessed category: B1 raises on purpose, and the
    Orchestrator's job is to carry that "we don't know" forward honestly, not to
    convert it into a confident-looking soft/hard.
    """
    try:
        return classify(channel, code), False
    except UnrecognizedDeclineCodeError:
        return None, True


def run_transaction(
    simulator: Simulator,
    billing_event: BillingEvent,
    customer: Customer,
    merchant: Merchant,
    runtime: SmartAgentRuntime,
    config: Optional[OrchestratorConfig] = None,
) -> TransactionResult:
    """Run the full recovery journey for one failed billing event
    (ARCHITECTURE.md D1, operations 1-11).

    The loop, per round: classify the decline that is being reacted to (B1) → ask the
    Circuit Breaker which channels survive (B4) → if none, hand to Human Fallback and
    stop (B5) → otherwise let the Bandit pick an arm (B2) → let Pacing say how long to
    wait (B3) → advance the clock → submit the attempt to the Simulator (A2) → record
    it and feed the outcome back into the Bandit's pool and Pacing's window → repeat.

    Two things about time are worth stating plainly, because they are easy to get
    silently wrong:

    - Every timestamp here is **simulated** time, from a `SimulatedClock` (A2 operation
      7). Nothing reads the wall clock.
    - That clock is **per transaction**, started at this billing event's scheduled time,
      rather than the Simulator's single global clock. A batch is processed one
      transaction at a time, so a shared clock would leave the second transaction
      starting weeks after its own billing event — and since the Simulator measures
      "has enough time passed to recover?" as elapsed time since the event was
      scheduled, every transaction after the first would appear instantly recoverable.
      `respond_to_attempt` takes `current_time` as an argument and never reads
      `simulator.clock`, so this costs nothing and keeps each journey honest.
    """
    config = config or OrchestratorConfig()

    transaction = TransactionState(
        transaction_id=f"{config.transaction_id_prefix}_{billing_event.billing_event_id}",
        billing_event_id=billing_event.billing_event_id,
        customer_id=customer.customer_id,
        merchant_id=customer.merchant_id,
    )
    result = TransactionResult(transaction=transaction)

    clock = SimulatedClock(billing_event.scheduled_at)
    horizon = billing_event.scheduled_at + timedelta(days=config.recovery_horizon_days)
    registered = [m.channel for m in customer.mandates]

    # Operation 1 — the decline this round reacts to. It starts as the *original*
    # billing-event failure, which by this project's convention is not itself an
    # attempt record (see notes/TRACKER.md's Phase 2 note): only the retries that
    # follow it are. It is carried as a `PendingDecline` so the Circuit Breaker still
    # sees it — without that, a card that hard-declined on the original charge would
    # look perfectly healthy to B4 on the first round.
    initial_code = simulator.first_attempt_code[billing_event.billing_event_id]
    initial_channel = customer.primary_channel()
    initial_category, unrecognized = _classify_decline(initial_channel, initial_code)

    if unrecognized or initial_category is None:
        # We cannot diagnose the failure we are supposed to be recovering from, so we
        # do not start guessing with someone's money — a human takes it from here. See
        # the same branch inside the loop below for the full reasoning.
        #
        # `closed_channels` is **empty**, and that is the honest value: the Circuit
        # Breaker closed nothing here. The loop stopped because B1 declined to
        # classify the code, not because anything was ruled out. This used to pass
        # `registered`, which read naturally but made the audit trail say the breaker
        # had permanently closed every channel on file — a safety mechanism credited
        # with a decision it never made (found in Phase 10 task 5; see notes/TRACKER.md).
        # The reason this transaction stopped travels on its closing line instead,
        # where the `unrecognized_decline_code` tag actually is.
        result.origin_channel = initial_channel
        result.origin_unrecognized = True
        result.human_fallback_event = run_human_fallback(
            transaction, clock.now(), merchant, closed_channels=[]
        )
        result.terminal_reason = "unrecognized_decline_code"
        return result

    # Two different things are tracked from here, and conflating them into one variable
    # was a real bug: the agent retried cards that had hard-declined on the original
    # charge (see notes/TRACKER.md's resolved Phase 8 finding).
    #
    # `origin_decline` is the original billing-event failure. It is set once and **never
    # reassigned**, because it is the only surviving record of that failure anywhere —
    # the original charge is deliberately not an `AttemptRecord` (the Phase 2
    # convention), so if this variable stops carrying it, nothing remembers it at all.
    # Every Circuit Breaker call below gets this one.
    origin_decline = circuit_breaker.PendingDecline(
        channel=initial_channel,
        category=initial_category,
        occurred_at=billing_event.scheduled_at,
    )
    result.origin_channel = initial_channel
    result.origin_category = initial_category

    # `context_decline` is whichever decline *this round* is reacting to — the original
    # failure on the first pass, then the most recent attempt's decline. It is what the
    # Bandit keys its learning on and what the audit trail records as this round's
    # context. It must move; `origin_decline` must not.
    #
    # Handing B4 the latest decline instead loses nothing, which is why this split is
    # safe: every retry decline is written to `transaction.attempts` before the next
    # round begins, so the Circuit Breaker already sees recent declines through the
    # attempt records. The original failure was the only thing it could not see for
    # itself.
    context_decline = origin_decline

    while not is_terminal(transaction):
        now = clock.now()

        # Operation 3 — ask the Circuit Breaker for each registered channel's status.
        # `channel_status` is the human-readable open/closed picture recorded for the
        # audit trail; `arms` is what the Bandit may actually choose from. They differ
        # for exactly one reason, documented in `bandit.available_arms`: card's 24h
        # spacing rule closes a channel *temporarily*, and a transaction must wait it
        # out rather than treat it as an option lost.
        channel_status = circuit_breaker.get_channel_status(
            transaction, registered, now, origin_decline
        )
        arms = available_arms(customer, transaction, now, origin_decline)

        # Operation 4 — no automated options left anywhere, so B5 ends the journey.
        if not arms:
            closed = [
                channel
                for channel in registered
                if circuit_breaker.is_channel_permanently_closed(
                    transaction, channel, origin_decline
                )
            ]
            result.human_fallback_event = run_human_fallback(
                transaction, now, merchant, closed_channels=closed
            )
            result.terminal_reason = "all_channels_closed"
            break

        if len(transaction.attempts) >= config.max_attempts_per_transaction:
            transaction.status = TransactionStatus.ABANDONED
            result.terminal_reason = "max_attempts_reached"
            break

        # Operation 5 — the Bandit chooses among the arms B4 still permits.
        context: BanditContext = context_key(
            context_decline.channel, context_decline.category
        )
        chosen, selection_mode = select_arm_with_mode(
            arms, context, runtime.stats_pool, runtime.rng, epsilon=config.epsilon
        )

        # Operation 6 — Pacing says when. AIMD reacts to the *system-wide* rolling
        # success rate (not this transaction's history), then jitter spreads out
        # transactions that would otherwise land on the same instant.
        rolling_success_rate = runtime.outcome_window.success_rate
        baseline_success_rate = runtime.outcome_window.baseline_success_rate
        health_signal = pacing.pacing_signal(runtime.outcome_window)
        aggressiveness = runtime.aimd.update(runtime.outcome_window)
        base_wait = runtime.aimd.base_wait_hours()
        jittered_wait = apply_jitter(base_wait, runtime.rng)

        # The Circuit Breaker's 24h card spacing rule is a floor under Pacing's number:
        # a safety rule may push an attempt later, never pull it earlier. Pacing stays
        # free to wait longer than the minimum whenever system health says it should.
        proposed_at = now + timedelta(hours=jittered_wait)
        attempt_at = circuit_breaker.earliest_next_attempt_at(
            transaction, chosen, proposed_at, origin_decline
        )
        spacing_applied = attempt_at > proposed_at

        if attempt_at > horizon:
            transaction.status = TransactionStatus.ABANDONED
            result.terminal_reason = "recovery_horizon_exceeded"
            break

        # Operation 7 — advance the (simulated) clock to the attempt's moment.
        clock.advance(attempt_at - now)

        # The authoritative gate: B4 has the final word immediately before any attempt
        # is submitted. Reaching this with a closed channel would mean the arm
        # filtering and the spacing floor above disagreed with the rule they are meant
        # to be enforcing, which is a sequencing bug in this file — so it fails loudly
        # rather than quietly making an attempt a real payment network would penalise.
        if not circuit_breaker.is_channel_open(
            transaction, chosen, attempt_at, origin_decline
        ):
            raise RuntimeError(
                f"Orchestrator was about to attempt {chosen.value!r} on "
                f"{transaction.transaction_id!r} at {attempt_at.isoformat()}, but the "
                "Circuit Breaker reports that channel closed."
            )

        # Operation 8 — submit the attempt to the Simulator (the "fake bank").
        # `route` stays None: route-level arms aren't modelled, per `ARM_SAME_CHANNEL`.
        outcome, decline_code, _simulated_category = simulator.respond_to_attempt(
            transaction, chosen, None, attempt_at
        )

        # Operation 2, for this attempt's own decline. The agent classifies the code
        # itself via B1 and deliberately ignores the category the Simulator returned
        # alongside it — that value is the fake bank's private knowledge, and letting
        # it leak into the decision path would quietly make the Classifier's accuracy
        # (a Phase 11 metric) a measurement of nothing.
        category: Optional[DeclineCategory] = None
        unrecognized = False
        if outcome == AttemptOutcome.DECLINED and decline_code is not None:
            category, unrecognized = _classify_decline(chosen, decline_code)

        # Operation 9 — record the attempt, then feed the outcome back into both
        # learning surfaces: the Bandit's shared pool and Pacing's rolling window.
        success = outcome == AttemptOutcome.APPROVED
        transaction.record_attempt(
            AttemptRecord(
                transaction_id=transaction.transaction_id,
                attempt_number=len(transaction.attempts) + 1,
                channel=chosen,
                route=None,
                attempted_at=attempt_at,
                chosen_arm=_arm_label(chosen, context_decline.channel),
                outcome=outcome,
                decline_code=decline_code,
                decline_category=category,
            )
        )
        runtime.stats_pool.record_outcome(context, chosen, success)
        runtime.outcome_window.record(success)

        # Operation 10 — the round's facts, for the Audit Trail (C1) to render later.
        result.decisions.append(
            AttemptDecision(
                transaction_id=transaction.transaction_id,
                attempt_number=len(transaction.attempts),
                decided_at=now,
                context_channel=context_decline.channel,
                context_category=context_decline.category,
                channel_status=tuple(channel_status.items()),
                available_arms=tuple(arms),
                chosen_channel=chosen,
                chosen_arm=_arm_label(chosen, context_decline.channel),
                selection_mode=selection_mode,
                rolling_success_rate=rolling_success_rate,
                baseline_success_rate=baseline_success_rate,
                health_signal=health_signal,
                aggressiveness=aggressiveness,
                base_wait_hours=base_wait,
                jittered_wait_hours=jittered_wait,
                actual_wait_hours=(attempt_at - now).total_seconds() / 3600.0,
                spacing_rule_applied=spacing_applied,
                attempted_at=attempt_at,
                outcome=outcome,
                decline_code=decline_code,
                decline_category=category,
                unrecognized_decline_code=unrecognized,
            )
        )

        # Operation 11 — approved ends the journey; declined sends us round again.
        if success:
            transaction.status = TransactionStatus.RECOVERED
            result.terminal_reason = "recovered"
            break

        if unrecognized or category is None:
            # B1 refused to guess a category for this code, so the Orchestrator refuses
            # to keep making money decisions on a diagnosis it doesn't have. Every
            # downstream step needs a category — the Bandit keys its learning on one,
            # and the Circuit Breaker's closure rules read one off the attempt record —
            # so continuing would mean inventing the very thing B1 declined to assert.
            # A human takes it from here, and the flag travels into the audit trail
            # (ARCHITECTURE.md C1 operation 5) rather than being smoothed over.
            result.human_fallback_event = run_human_fallback(
                transaction, attempt_at, merchant, closed_channels=[]
            )
            result.terminal_reason = "unrecognized_decline_code"
            break

        # Only the *context* moves on. `origin_decline` stays exactly as it was — that
        # is the whole point of the split, and reassigning it here is the bug this file
        # used to have.
        context_decline = circuit_breaker.PendingDecline(
            channel=chosen, category=category, occurred_at=attempt_at
        )

    return result


def iter_batch(
    simulator: Simulator,
    runtime: Optional[SmartAgentRuntime] = None,
    config: Optional[OrchestratorConfig] = None,
) -> Iterator[TransactionResult]:
    """Run the Smart Agent over every failed billing event in a seeded Simulator batch,
    yielding each transaction the moment it finishes (ARCHITECTURE.md D1, operation 12 —
    the Smart Agent half).

    Billing events that never failed are skipped: there is nothing to recover, and
    manufacturing a transaction for them would inflate the recovery rate the Metrics
    engine (C3) reports with cases that were never at risk.

    One `SmartAgentRuntime` is shared across the whole batch on purpose — see that
    class's docstring — so the Bandit genuinely learns across customers and merchants,
    and AIMD genuinely reacts to system-wide health, as the batch progresses.

    **Why this is a generator, and `run_batch` below is the thin wrapper.** The Dashboard
    (C4, operation 1) narrates a batch *while it runs*, which a function that only
    returns once every transaction is finished cannot support. Yielding per transaction
    is the smallest change that gives the live feed something to watch, and it deliberately
    does not give the caller a way to alter the run: a consumer can observe each result,
    not steer the next one. The batch filter above stays in this one place, which matters
    because both agents must apply it identically or their denominators stop matching.
    """
    config = config or OrchestratorConfig()
    runtime = runtime or SmartAgentRuntime.from_config(config)

    customers_by_id = {c.customer_id: c for c in simulator.customers}
    merchants_by_id = {m.merchant_id: m for m in simulator.merchants}

    for billing_event in simulator.billing_events:
        if billing_event.billing_event_id not in simulator.failed_billing_event_ids:
            continue
        customer = customers_by_id[billing_event.customer_id]
        merchant = merchants_by_id[billing_event.merchant_id]
        yield run_transaction(
            simulator, billing_event, customer, merchant, runtime, config
        )


def run_batch(
    simulator: Simulator,
    runtime: Optional[SmartAgentRuntime] = None,
    config: Optional[OrchestratorConfig] = None,
) -> list[TransactionResult]:
    """The whole batch, run to completion — `iter_batch` drained into a list.

    Unchanged in behaviour and signature from the form every phase since Phase 8 has
    called; the loop simply moved one function up so the Dashboard can also watch it
    happen.
    """
    return list(iter_batch(simulator, runtime, config))
