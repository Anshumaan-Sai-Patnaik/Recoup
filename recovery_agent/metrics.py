"""C3 — Metrics & Evaluation Engine.

The component that turns two runs' worth of raw records into the numbers that actually
prove something. Everything before this point *behaves*; this is where the behaviour is
measured.

It answers the two questions ARCHITECTURE.md C3 sets out. The **head-to-head half**
(operations 1-4) compares the Smart Agent to the Baseline Agent over one identical
seeded world. The **against-hidden-truth half** (operations 5-7) compares an agent to
what was actually possible, using the answer key the Simulator has held privately since
the world was generated. `evaluate()` then packages both, together with the exact
configuration that produced them, into the one citable object anybody should quote
(operation 8).

Two rules shape every function below, and both are worth stating before the code:

**1. Measure what was recorded; never re-run a decision to find out what it was.**
Every number here is derived from facts the agents wrote down while they ran —
`TransactionState.status`, the attempt list, each `AttemptDecision`, the terminal reason
tag. Nothing in this file calls the Circuit Breaker, the Classifier or the Simulator to
reconstruct what "would have" happened. A measurement that re-derives its subject is
measuring this file's copy of the rule, not the run.

**2. A number the Baseline never generated is `None`, not zero.**
The Baseline Agent has no Circuit Breaker, so it closed zero channels — but writing `0`
in that cell would say it *checked* and found nothing to close, which is false, and
would let a reader compare 43 against 0 as if both agents had played the same game. The
absence is the finding (it is why the Baseline burns attempts on dead channels), so it
is reported as an explicit "not applicable, this agent never consulted the mechanism".
This is the same honesty rule the Audit Trail (C1) enforces on sentences, applied to
cells in a table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import pandas as pd

from recovery_agent.audit_trail import (
    AGENT_BASELINE,
    AGENT_SMART,
    AgentResult,
    agent_for,
    write_text,
)
from recovery_agent.baseline_agent import BaselineConfig, PairedRun, run_paired_batch
from recovery_agent.models import DeclineCategory, MandateChannel, TransactionStatus
from recovery_agent.orchestrator import OrchestratorConfig
from recovery_agent.simulator import Simulator, SimulatorConfig

# Which agents actually consult the Circuit Breaker (B4). The Smart Agent does; the
# Baseline deliberately does not, which is half of what the experiment is testing.
# Kept as one mapping rather than scattered `if agent == ...` checks so that the "not
# applicable" cells in the comparison all come from a single stated fact.
_CONSULTS_CIRCUIT_BREAKER = {AGENT_SMART: True, AGENT_BASELINE: False}


def consults_circuit_breaker(agent: str) -> bool:
    """Whether `agent` ever asked the Circuit Breaker anything during its run.

    Drives every "not applicable" in the closure metrics. Raises on an unknown agent
    label rather than defaulting to `False`, because silently reporting a new agent's
    closures as "never consulted" would be a lie about a mechanism it might well have.
    """
    try:
        return _CONSULTS_CIRCUIT_BREAKER[agent]
    except KeyError:
        raise ValueError(
            f"Unknown agent label {agent!r}: expected {AGENT_SMART!r} or "
            f"{AGENT_BASELINE!r}."
        ) from None


# ---------------------------------------------------------------------------
# Per-transaction facts
# ---------------------------------------------------------------------------


def permanently_closed_channels(
    result: AgentResult,
) -> Optional[tuple[MandateChannel, ...]]:
    """Which of this customer's channels the Circuit Breaker retired for good during
    this transaction's journey, in the order they were first observed closed.

    Returns `None` — not an empty tuple — for an agent that never consults B4. Empty
    means "the breaker was asked and closed nothing"; `None` means "nobody asked".

    **Read off the record, not recomputed.** Each `AttemptDecision` already carries both
    halves of the answer: `channel_status` lists every channel the customer had
    registered, and `available_arms` lists those the Bandit was allowed to pick from,
    which `bandit.available_arms` defines as exactly the registered channels that are
    not *permanently* closed. The difference between the two, at the moment the decision
    was made, is therefore the set of channels B4 had walled off — with no need to call
    B4 again here, and no risk of this file's idea of the rules drifting from the ones
    the run actually enforced.

    The permanent/temporary distinction matters and is not cosmetic. A card inside its
    24h spacing window shows as `"closed"` in `channel_status` while still being a
    perfectly live option a day later; counting those as closures would inflate this
    number several-fold and describe a wait as a wall.

    The final round is picked up separately from `human_fallback_event.closed_channels`,
    because when every channel is permanently closed the loop escalates immediately and
    emits no `AttemptDecision` for that round — so the closures that ended the journey
    exist only on the fallback event.
    """
    if not consults_circuit_breaker(agent_for(result)):
        return None

    ordered: list[MandateChannel] = []
    seen: set[MandateChannel] = set()

    def note(channel: MandateChannel) -> None:
        if channel not in seen:
            seen.add(channel)
            ordered.append(channel)

    for decision in result.decisions:
        arms = set(getattr(decision, "available_arms", ()))
        for channel, _status in getattr(decision, "channel_status", ()):
            if channel not in arms:
                note(channel)

    if result.human_fallback_event is not None:
        for channel in result.human_fallback_event.closed_channels:
            note(channel)

    return tuple(ordered)


def channels_used(result: AgentResult) -> tuple[MandateChannel, ...]:
    """The distinct channels this transaction actually attempted, in first-use order.

    Reads the attempt records rather than the decisions, so it counts attempts that
    were really submitted rather than choices that were made.
    """
    ordered: list[MandateChannel] = []
    seen: set[MandateChannel] = set()
    for attempt in result.transaction.attempts:
        if attempt.channel not in seen:
            seen.add(attempt.channel)
            ordered.append(attempt.channel)
    return tuple(ordered)


def transaction_row(result: AgentResult) -> dict[str, Any]:
    """One finished transaction flattened into the row the aggregations below group on.

    Deliberately one row *per transaction*, not per attempt: every operation C3 asks for
    in this task is a per-transaction question ("did it recover?", "how many attempts did
    it take?", "was a human asked?"), and aggregating those from an attempt-level frame
    would mean grouping and de-duplicating on every single call.

    Both agents' result types flatten through this one function — they carry the same
    four fields and differ only in the richness of their decision records — which is what
    makes the head-to-head a genuine like-for-like comparison rather than two separately
    computed tables placed side by side.
    """
    agent = agent_for(result)
    transaction = result.transaction
    closed = permanently_closed_channels(result)
    used = channels_used(result)

    return {
        "agent": agent,
        "transaction_id": transaction.transaction_id,
        "billing_event_id": transaction.billing_event_id,
        "customer_id": transaction.customer_id,
        "merchant_id": transaction.merchant_id,
        "status": transaction.status.value,
        "terminal_reason": result.terminal_reason,
        "attempt_count": len(transaction.attempts),
        "recovered": transaction.status == TransactionStatus.RECOVERED,
        "escalated_to_human": transaction.status == TransactionStatus.ESCALATED_TO_HUMAN,
        "abandoned": transaction.status == TransactionStatus.ABANDONED,
        "unresolved": transaction.status == TransactionStatus.IN_PROGRESS,
        "channels_used": ",".join(c.value for c in used),
        "channel_count": len(used),
        # None (not 0, not "") wherever the breaker was never consulted — see this
        # module's docstring, rule 2.
        "circuit_breaker_consulted": consults_circuit_breaker(agent),
        "channels_permanently_closed": (
            ",".join(c.value for c in closed) if closed is not None else None
        ),
        "closure_count": len(closed) if closed is not None else None,
    }


def transaction_frame(results: Iterable[AgentResult]) -> pd.DataFrame:
    """One agent's finished batch as a `pandas` frame, one row per transaction.

    The single tabular surface every aggregation in this file works from, and the object
    the Dashboard (C4) can hand straight to a table widget or a chart.

    `dtype=object` is *not* used here, unlike the Audit Trail's export: these rows all
    have the same keys, so pandas' inference has no gaps to widen an integer column
    over — except `closure_count`, which is genuinely absent for the Baseline. That one
    is nullable-integer (`Int64`) so a missing closure count stays missing instead of
    turning the column into floats and printing "43.0 closures".
    """
    rows = [transaction_row(result) for result in results]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["closure_count"] = frame["closure_count"].astype("Int64")
    return frame


# ---------------------------------------------------------------------------
# Head-to-head summary (ARCHITECTURE.md C3, operations 1-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSummary:
    """One agent's headline numbers over one batch.

    Frozen, like every record in this project: a results object that can be edited after
    the fact is not a result. The four fields ARCHITECTURE.md C3 operations 1-4 actually
    ask for are `recovery_rate`, `channel_closures`, `escalated_to_human` and
    `mean_attempts_to_recovery`; the rest are the raw counts those are computed from,
    kept alongside so a reader can check the arithmetic and so no consumer has to
    recompute a denominator.
    """

    agent: str

    transactions: int
    """Failed billing events this agent was asked to recover. Identical across both
    agents by construction (`baseline_agent.run_paired_batch`), which is what makes
    every rate below directly comparable."""

    recovered: int
    escalated_to_human: int
    abandoned: int
    unresolved: int
    """Transactions still `IN_PROGRESS` at the end of the run. Should always be 0 — both
    loops run to a terminal state — and is reported rather than assumed so that a run
    which silently left journeys open shows it here instead of quietly shrinking the
    denominator of every rate."""

    recovery_rate: float
    """Operation 1, the headline metric: recovered / transactions."""

    escalation_rate: float
    abandonment_rate: float

    total_attempts: int
    """Every retry this agent submitted across the batch. Not itself one of the four
    operations, but the cost side of the recovery rate: a higher recovery rate bought
    with more attempts on the payment network is a different result from the same rate
    bought with fewer, and the comparison is misleading without it."""

    mean_attempts_all: float

    mean_attempts_to_recovery: Optional[float]
    """Operation 4, computed **among recovered transactions only** — the average number
    of attempts a success took. `None` if nothing recovered, rather than 0, which would
    read as "recoveries were free"."""

    median_attempts_to_recovery: Optional[float]

    circuit_breaker_consulted: bool

    channel_closures: Optional[int]
    """Operation 2: how many (transaction, channel) pairs the Circuit Breaker closed
    permanently across the batch. `None` for an agent that has no Circuit Breaker — see
    this module's docstring, rule 2."""

    transactions_with_a_closure: Optional[int]
    """How many distinct transactions lost at least one channel. Reported next to
    `channel_closures` because the two answer different questions: one transaction
    losing three channels and three transactions losing one each are the same closure
    count and very different situations."""

    closures_by_channel: Optional[dict[str, int]]

    terminal_reasons: dict[str, int]
    """The agents' own machine-readable reason tags and their counts.

    Kept because the same *status* is reached by different *routes* in the two agents,
    and the status counts alone hide that: the Smart Agent escalates on
    `all_channels_closed` (it ran out of options), while the Baseline escalates on
    `attempt_cap_reached` (a counter ran out). Both are honestly "a human was asked",
    which is what makes operation 3 comparable, but a reader drawing conclusions about
    *why* needs this column and not the status one.
    """

    def to_dict(self) -> dict[str, Any]:
        """A flat, JSON/CSV-friendly row, matching the convention every event type in
        this project already follows. The two dict-valued fields are kept as dicts here
        rather than flattened into columns, because their keys differ per agent and per
        run; `HeadToHead.to_frame` renders them as text where a flat cell is needed."""
        return {
            "agent": self.agent,
            "transactions": self.transactions,
            "recovered": self.recovered,
            "escalated_to_human": self.escalated_to_human,
            "abandoned": self.abandoned,
            "unresolved": self.unresolved,
            "recovery_rate": self.recovery_rate,
            "escalation_rate": self.escalation_rate,
            "abandonment_rate": self.abandonment_rate,
            "total_attempts": self.total_attempts,
            "mean_attempts_all": self.mean_attempts_all,
            "mean_attempts_to_recovery": self.mean_attempts_to_recovery,
            "median_attempts_to_recovery": self.median_attempts_to_recovery,
            "circuit_breaker_consulted": self.circuit_breaker_consulted,
            "channel_closures": self.channel_closures,
            "transactions_with_a_closure": self.transactions_with_a_closure,
            "closures_by_channel": self.closures_by_channel,
            "terminal_reasons": self.terminal_reasons,
        }


def _rate(count: int, total: int) -> float:
    """A proportion that refuses to divide by zero. An empty batch has no recovery rate
    at all, and 0.0 is the least misleading stand-in for "no transactions" here — every
    count it is built from is also 0, so nothing reads as a real result."""
    return count / total if total else 0.0


def _optional_mean(values: "pd.Series[Any]") -> Optional[float]:
    """The mean of a possibly-empty series, as `None` rather than `nan` when empty.

    `nan` propagates silently through arithmetic and formats as "nan" in a dashboard;
    `None` forces a caller to decide what "no recoveries happened" should look like.
    """
    return float(values.mean()) if len(values) else None


def summarise(
    results: Sequence[AgentResult], frame: Optional[pd.DataFrame] = None
) -> AgentSummary:
    """One agent's batch reduced to its headline numbers (C3, operations 1-4).

    `frame` is an optional pre-built `transaction_frame` — pass it when you already have
    one (the paired comparison does) so a batch isn't flattened twice.

    Every aggregation runs through `pandas`, per DESIGN.md's mapping for this component:
    grouping and counting tabular records is what it is for, and hand-rolled loops here
    would be a second implementation of arithmetic pandas already does correctly.
    """
    if not results:
        raise ValueError(
            "Cannot summarise an empty batch: there is no agent to attribute the "
            "result to, and no denominator for any rate."
        )

    agent = agent_for(results[0])
    mixed = {agent_for(result) for result in results} - {agent}
    if mixed:
        raise ValueError(
            f"summarise() expects one agent's batch; got {agent!r} mixed with "
            f"{sorted(mixed)!r}. Summarise each agent separately and compare the two "
            "with head_to_head()."
        )

    frame = transaction_frame(results) if frame is None else frame
    total = len(frame)

    recovered_attempts = frame.loc[frame["recovered"], "attempt_count"]
    recovered = int(frame["recovered"].sum())
    escalated = int(frame["escalated_to_human"].sum())
    abandoned = int(frame["abandoned"].sum())
    unresolved = int(frame["unresolved"].sum())

    consulted = consults_circuit_breaker(agent)
    if consulted:
        closures = int(frame["closure_count"].sum())
        with_closure = int((frame["closure_count"] > 0).sum())
        by_channel: Optional[dict[str, int]] = _closures_by_channel(frame)
    else:
        closures = None
        with_closure = None
        by_channel = None

    return AgentSummary(
        agent=agent,
        transactions=total,
        recovered=recovered,
        escalated_to_human=escalated,
        abandoned=abandoned,
        unresolved=unresolved,
        recovery_rate=_rate(recovered, total),
        escalation_rate=_rate(escalated, total),
        abandonment_rate=_rate(abandoned, total),
        total_attempts=int(frame["attempt_count"].sum()),
        mean_attempts_all=float(frame["attempt_count"].mean()) if total else 0.0,
        mean_attempts_to_recovery=_optional_mean(recovered_attempts),
        median_attempts_to_recovery=(
            float(recovered_attempts.median()) if len(recovered_attempts) else None
        ),
        circuit_breaker_consulted=consulted,
        channel_closures=closures,
        transactions_with_a_closure=with_closure,
        closures_by_channel=by_channel,
        terminal_reasons=_value_counts(frame["terminal_reason"]),
    )


def _value_counts(column: "pd.Series[Any]") -> dict[str, int]:
    """A column's value counts as a plain dict, ordered most common first."""
    return {str(key): int(count) for key, count in column.value_counts().items()}


def _closures_by_channel(frame: pd.DataFrame) -> dict[str, int]:
    """How many transactions each channel was permanently closed on.

    Explodes the comma-joined `channels_permanently_closed` cell back into one row per
    (transaction, channel) pair and counts them, so the total across this dict equals
    `channel_closures` exactly.
    """
    exploded = (
        frame["channels_permanently_closed"]
        .fillna("")
        .str.split(",")
        .explode()
        .str.strip()
    )
    return _value_counts(exploded[exploded != ""])


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadToHead:
    """Both agents' headline numbers over one identical seeded world.

    This is ARCHITECTURE.md C3 operations 1-4 as a single object: the controlled
    experiment `baseline_agent.run_paired_batch` sets up, finally scored. It holds no
    seed or configuration of its own yet — packaging a *citable, reproducible* result
    (operation 8) is a later task in this phase, and stamping a half-built object with a
    seed would invite quoting it as final before the against-truth half exists.
    """

    smart: AgentSummary
    baseline: AgentSummary

    @property
    def recovery_rate_gain_points(self) -> float:
        """The headline: percentage *points* of recovery rate the Smart Agent adds over
        the Baseline. Points, not percent, because the two are routinely confused and
        the difference flatters whichever one sounds larger."""
        return (self.smart.recovery_rate - self.baseline.recovery_rate) * 100.0

    @property
    def attempts_saved(self) -> int:
        """How many fewer (or, if negative, more) attempts the Smart Agent submitted to
        recover the same batch."""
        return self.baseline.total_attempts - self.smart.total_attempts

    @property
    def attempt_reduction(self) -> float:
        """`attempts_saved` as a fraction of the Baseline's attempt count."""
        return _rate(self.attempts_saved, self.baseline.total_attempts)

    def to_frame(self) -> pd.DataFrame:
        """The comparison as a two-column table, one metric per row.

        Metric-per-row rather than agent-per-row because that is how it gets read — a
        judge scans down the metric names comparing two numbers side by side, and a
        dashboard renders it without transposing. Cells that are genuinely not
        applicable carry the words `"not applicable (no circuit breaker)"` rather than a
        blank or a zero, so the table states the reason itself instead of relying on a
        caption nobody reads.
        """
        smart = self.smart.to_dict()
        baseline = self.baseline.to_dict()
        rows = []
        for key in smart:
            if key == "agent":
                continue
            rows.append(
                {
                    "metric": key,
                    AGENT_SMART: _cell(smart[key], self.smart),
                    AGENT_BASELINE: _cell(baseline[key], self.baseline),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Both summaries plus the three derived comparisons, JSON-friendly."""
        return {
            AGENT_SMART: self.smart.to_dict(),
            AGENT_BASELINE: self.baseline.to_dict(),
            "recovery_rate_gain_points": self.recovery_rate_gain_points,
            "attempts_saved": self.attempts_saved,
            "attempt_reduction": self.attempt_reduction,
        }


NOT_APPLICABLE_NO_BREAKER = "not applicable (no circuit breaker)"
"""What a closure cell says for an agent that never consults B4.

Spelled out in the cell itself, in the same spirit as the Audit Trail's sentences: the
table has to be readable on its own, and "0" next to "43" would be read as a comparison
between two agents that both counted, which is exactly the wrong conclusion.
"""


NOT_APPLICABLE_NO_CLASSIFIER = "not applicable (no classifier)"
"""What an operation-7 cell says for an agent that never consults B1.

The exact counterpart of `NOT_APPLICABLE_NO_BREAKER`, and it exists for the same reason:
`AgainstTruth.classifier` is `None` for the Baseline because it never formed an opinion
about *why* a payment failed — not because it formed one and scored zero. Rendering that
absence as `0`, or as an empty cell a reader fills in themselves, would credit the Smart
Agent with beating a diagnosis nobody ever made.
"""


def _cell(value: Any, summary: AgentSummary) -> Any:
    """Render one summary field for the comparison table."""
    if value is None:
        return (
            NOT_APPLICABLE_NO_BREAKER
            if not summary.circuit_breaker_consulted
            else "none"
        )
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items()) or "none"
    return value


def head_to_head(paired: PairedRun) -> HeadToHead:
    """Score both agents over one paired run (ARCHITECTURE.md C3, operations 1-4).

    Takes the `PairedRun` rather than two loose result lists on purpose: that object is
    the guarantee that both agents faced the same customers, the same billing events and
    the same hidden truths, and requiring it here means a head-to-head comparison cannot
    accidentally be built from two unrelated batches.
    """
    return HeadToHead(
        smart=summarise(paired.smart),
        baseline=summarise(paired.baseline),
    )


# ---------------------------------------------------------------------------
# Against the hidden truth (ARCHITECTURE.md C3, operations 5-7)
# ---------------------------------------------------------------------------
#
# The second, harder question. The head-to-head above says which agent did better; this
# says how either of them did against what was *actually possible*, using the answer key
# the Simulator has held privately since the world was generated and that no decision-
# making component has ever been allowed to see.
#
# ARCHITECTURE.md scopes operations 5-7 as a Smart Agent diagnostic. `against_truth`
# nevertheless works for either agent, and the results object computes it for both,
# because the waste metric (operation 6) is close to meaningless without a reference
# point: "7.6 attempts per hopeless transaction" only becomes a finding next to the
# Baseline's 10.0. Operation 7 has no Baseline counterpart at all and reports as
# not-applicable there, by the same rule that governs the closure metrics above.


def _consults_classifier(agent: str) -> bool:
    """Whether `agent` ever asked the Classifier (B1) anything.

    The Smart Agent opens every journey with a diagnosis; the Baseline retries without
    ever forming an opinion about what went wrong. That absence is why operation 7's
    numbers are `None` for it rather than 0 or 100%.
    """
    return consults_circuit_breaker(agent)


@dataclass(frozen=True)
class ClassifierDiagnosis:
    """ARCHITECTURE.md C3 operation 7 — what the Classifier's opening soft/hard call was
    actually worth, checked against the Hidden Truth.

    **Read the two halves of this differently**, because the obvious reading of
    "classifier accuracy" is wrong here in two separate ways, and quoting it as a
    performance number would be the kind of dishonesty this project's audit trail exists
    to prevent.

    *First*, `label_agreement` compares B1's call against the category the Simulator
    itself rolled for that failure. It is **1.0 by construction, and that is not an
    achievement**: the Simulator picks a decline code *because* it has already decided
    the failure is soft or hard, drawing from the same JSON registries B1 later reads. It
    could not disagree without one of the two being broken. It is kept as a **regression
    canary** — a registry edit that miscategorised a code, or a simulator that started
    emitting codes off its own diagnosis, would show up here as a number below 1.0 and
    nowhere else — and it is labelled as one, never as evidence the classifier works.

    *Second*, and more usefully, the soft/hard call is **not a prediction of
    recoverability**, so scoring it as one manufactures errors that were never made:

    - A *soft* decline means "temporary, worth another try", explicitly a **maybe**
      (IDEA.md §7). This world makes soft failures genuinely recoverable 85% of the
      time; the other 15% are not classifier mistakes, because the classifier never
      claimed a recovery.
    - A *hard* decline means "this instrument is permanently dead", a claim about **the
      channel**, not about the transaction. A hard-declined customer can still be
      recovered by switching to another mandate, and about half of them are. Counting
      those as classifier errors would be a false accusation of the one component that
      got it right.

    So the recoverability figures below are reported as **conditional base rates**, which
    is what they are, and the hard call's own claim is scored separately and honestly
    against the only thing it actually asserted: that the channel it was made on was
    dead.
    """

    calls: int
    """Transactions whose opening diagnosis was recorded. Should equal the batch size."""

    soft_calls: int
    hard_calls: int
    unrecognized_calls: int
    """Opening codes B1 refused to classify. Zero in an unmodified run — the Simulator
    only emits codes that are in the registries — and non-zero the moment somebody edits
    a registry, which is exactly when a reader needs to see it rather than have it folded
    silently into the soft or hard column."""

    label_agreement: Optional[float]
    """Agreement with the Simulator's own category for the same failure. See the class
    docstring: **1.0 by construction**, kept as a regression canary, never quoted as
    accuracy."""

    label_disagreements: int

    recoverable_given_soft: Optional[float]
    """Of the failures B1 called soft, the fraction the answer key says were recoverable
    *somewhere*. A base rate, not an accuracy."""

    recoverable_given_hard: Optional[float]
    """The same for hard calls — recoverable only ever by switching channel, since a hard
    decline kills the channel it was made on."""

    hard_calls_on_a_dead_channel: int
    """Hard calls where the answer key agrees the channel really was finished: the
    primary channel is absent from the truth's recoverable channels."""

    hard_call_channel_precision: Optional[float]
    """`hard_calls_on_a_dead_channel / hard_calls` — the hard call scored against what it
    actually asserted. Also 1.0 by construction in this world (the Simulator never lets a
    hard-declined primary channel recover), and stated as such."""

    attempts_on_a_dead_origin_channel: int
    """The behavioural consequence, and the number the project's whole premise turns on:
    retries the agent spent on a channel its own opening diagnosis had called permanently
    dead. IDEA.md §4 names this as the single behaviour this project exists to prevent,
    so anything other than 0 for the Smart Agent is a defect, not a metric.

    Computed from the agent's own recorded call, not from the answer key — it measures
    whether the agent acted on what it knew, which is a fair test; whether it should have
    known more is what the rest of this class is for.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "soft_calls": self.soft_calls,
            "hard_calls": self.hard_calls,
            "unrecognized_calls": self.unrecognized_calls,
            "label_agreement": self.label_agreement,
            "label_disagreements": self.label_disagreements,
            "recoverable_given_soft": self.recoverable_given_soft,
            "recoverable_given_hard": self.recoverable_given_hard,
            "hard_calls_on_a_dead_channel": self.hard_calls_on_a_dead_channel,
            "hard_call_channel_precision": self.hard_call_channel_precision,
            "attempts_on_a_dead_origin_channel": self.attempts_on_a_dead_origin_channel,
        }


@dataclass(frozen=True)
class AgainstTruth:
    """One agent scored against the Simulator's private answer key (C3, operations 5-7).

    Frozen, like every other record here. Everything in it is derived from two sources
    that never touched each other during the run: what the agent recorded, and what the
    Simulator decided before the agent started.
    """

    agent: str

    transactions: int
    scored: int
    """Transactions that had a Hidden Truth to score against. Reported separately from
    `transactions` so that a batch where some journey had no answer key shrinks a visible
    number rather than a silent denominator."""

    recoverable: int
    unrecoverable: int

    # Operation 5 — recall on the genuinely recoverable.
    recovered_of_recoverable: int
    recall: Optional[float]
    """Operation 5, the diagnostic headline: of the failures that *could* have been
    recovered, the fraction this agent actually recovered. The one number that separates
    "did well" from "did as well as was possible"."""

    missed_recoverable: int
    missed_by_terminal_reason: dict[str, int]
    """Why the recoverable ones that got away, got away — the agent's own reason tags.
    A miss because the agent ran out of *time* (the horizon) is a different problem from
    a miss because the Circuit Breaker closed everything, and only this breakdown
    distinguishes them."""

    recovered_of_unrecoverable: int
    """Recoveries the answer key says were impossible. **Must be 0.** It is a consistency
    check on the world rather than a measure of the agent: the Simulator only ever
    approves an attempt its Hidden Truth sanctions, so a non-zero here means the answer
    key and the fake bank have disagreed and every other number on this page is suspect.
    Reported rather than asserted, so that failure is visible instead of assumed away."""

    # Operation 6 — waste on the genuinely unrecoverable.
    wasted_attempts: int
    """Every retry spent on a transaction the answer key says was never going to recover.

    "Wasted" is hindsight, and the distinction matters when quoting this: the agent could
    not have known, and a *reasonable* number of attempts on a failure that looks
    temporary is correct behaviour, not error. What this measures is how quickly an agent
    stops paying for a lost cause, which is the Circuit Breaker's and the horizon's job.
    """

    mean_wasted_attempts: Optional[float]
    max_wasted_attempts: Optional[int]
    wasted_attempt_share: Optional[float]
    """Wasted attempts as a fraction of every attempt the agent made. The cost of being
    wrong, as a share of total effort.

    **The one number in this file that reads backwards, so quote it carefully.** On the
    standard test world the Smart Agent's share is *higher* than the Baseline's (55%
    against 45%) while its absolute waste is *lower* (381 attempts against 500). Both
    are true and neither is a bug: the Smart Agent cut its total attempts by a third,
    and most of what it cut was on the recoverable side, where it now succeeds in two or
    three tries instead of five. Shrinking the denominator faster than the numerator
    raises the ratio. What is left is a genuine limit worth naming rather than hiding:
    against a failure that merely looks temporary forever, the agent has no signal
    telling it to stop early, so it keeps paying until the Circuit Breaker's caps or the
    horizon end the journey. That is what a Marginal Value Theorem layer (IDEA.md §8d,
    deferred as E1) would be for. Cite `wasted_attempts` and `mean_wasted_attempts` for
    the comparison; cite this one only alongside them.
    """

    # Operation 7 — the Classifier's own call.
    classifier: Optional[ClassifierDiagnosis]
    """`None` for an agent with no Classifier. Not an empty diagnosis and not a score of
    zero: the Baseline never formed an opinion to be right or wrong about."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "transactions": self.transactions,
            "scored": self.scored,
            "recoverable": self.recoverable,
            "unrecoverable": self.unrecoverable,
            "recovered_of_recoverable": self.recovered_of_recoverable,
            "recall": self.recall,
            "missed_recoverable": self.missed_recoverable,
            "missed_by_terminal_reason": self.missed_by_terminal_reason,
            "recovered_of_unrecoverable": self.recovered_of_unrecoverable,
            "wasted_attempts": self.wasted_attempts,
            "mean_wasted_attempts": self.mean_wasted_attempts,
            "max_wasted_attempts": self.max_wasted_attempts,
            "wasted_attempt_share": self.wasted_attempt_share,
            "classifier": (
                self.classifier.to_dict() if self.classifier is not None else None
            ),
        }


def truth_rows(
    results: Iterable[AgentResult], simulator: Simulator
) -> list[dict[str, Any]]:
    """Each transaction paired with the answer key for its billing event.

    The join every operation below aggregates over, and the **only** place in this
    project where an agent's output and the Simulator's private Hidden Truth are put side
    by side. That isolation is deliberate: the truth is meant to be invisible everywhere
    except here, and keeping the join in one function makes it obvious that no decision
    path can reach it.

    The customer index is rebuilt from `simulator.customers` rather than reaching into
    the Simulator's private lookup, so this file depends only on the Simulator's public
    surface.
    """
    customers = {c.customer_id: c for c in simulator.customers}
    rows: list[dict[str, Any]] = []

    for result in results:
        transaction = result.transaction
        truth = simulator.hidden_truths.get(transaction.billing_event_id)
        customer = customers.get(transaction.customer_id)
        primary = customer.primary_channel() if customer is not None else None

        # The agent's *own* opening diagnosis, read off what it recorded rather than
        # recomputed here (see this module's rule 1). The Baseline records none, because
        # it never made one.
        origin_channel = getattr(result, "origin_channel", None)
        origin_category = getattr(result, "origin_category", None)
        origin_unrecognized = bool(getattr(result, "origin_unrecognized", False))

        recoverable_channels = list(truth.recoverable_channels) if truth else []
        rows.append(
            {
                "agent": agent_for(result),
                "transaction_id": transaction.transaction_id,
                "billing_event_id": transaction.billing_event_id,
                "status": transaction.status.value,
                "terminal_reason": result.terminal_reason,
                "attempt_count": len(transaction.attempts),
                "recovered": transaction.status == TransactionStatus.RECOVERED,
                "has_truth": truth is not None,
                "is_recoverable": truth.is_recoverable if truth else None,
                "recoverable_channels": ",".join(c.value for c in recoverable_channels),
                # Whether the answer key left the customer's *primary* channel alive.
                # This is what a hard call actually asserted the opposite of.
                "primary_channel_alive": (
                    primary in recoverable_channels if truth is not None else None
                ),
                "simulated_category": (
                    simulator.first_attempt_category.get(
                        transaction.billing_event_id
                    ).value
                    if simulator.first_attempt_category.get(transaction.billing_event_id)
                    else None
                ),
                "origin_category": (
                    origin_category.value if origin_category is not None else None
                ),
                "origin_unrecognized": origin_unrecognized,
                "attempts_on_origin_channel": (
                    transaction.channel_attempt_count(origin_channel)
                    if origin_channel is not None
                    else 0
                ),
            }
        )
    return rows


def truth_frame(results: Iterable[AgentResult], simulator: Simulator) -> pd.DataFrame:
    """`truth_rows` as a `pandas` frame — the per-transaction table the against-truth
    numbers aggregate over, and a useful export in its own right for anyone who wants to
    inspect individual misses rather than read a summary."""
    return pd.DataFrame(truth_rows(results, simulator))


def _classifier_diagnosis(frame: pd.DataFrame) -> ClassifierDiagnosis:
    """Operation 7, computed over the scored rows. See `ClassifierDiagnosis` for why the
    recoverability figures are reported as base rates rather than as accuracy."""
    soft = frame[frame["origin_category"] == DeclineCategory.SOFT.value]
    hard = frame[frame["origin_category"] == DeclineCategory.HARD.value]
    unrecognized = frame[frame["origin_unrecognized"]]

    labelled = frame[frame["origin_category"].notna() & frame["simulated_category"].notna()]
    agreements = int((labelled["origin_category"] == labelled["simulated_category"]).sum())

    hard_on_dead = int((~hard["primary_channel_alive"].astype(bool)).sum())

    return ClassifierDiagnosis(
        calls=int(len(frame)),
        soft_calls=int(len(soft)),
        hard_calls=int(len(hard)),
        unrecognized_calls=int(len(unrecognized)),
        label_agreement=(agreements / len(labelled)) if len(labelled) else None,
        label_disagreements=int(len(labelled) - agreements),
        recoverable_given_soft=(
            float(soft["is_recoverable"].astype(bool).mean()) if len(soft) else None
        ),
        recoverable_given_hard=(
            float(hard["is_recoverable"].astype(bool).mean()) if len(hard) else None
        ),
        hard_calls_on_a_dead_channel=hard_on_dead,
        hard_call_channel_precision=(hard_on_dead / len(hard)) if len(hard) else None,
        attempts_on_a_dead_origin_channel=int(hard["attempts_on_origin_channel"].sum()),
    )


def against_truth(
    results: Sequence[AgentResult],
    simulator: Simulator,
    frame: Optional[pd.DataFrame] = None,
) -> AgainstTruth:
    """Score one agent's batch against the Simulator's private answer key
    (ARCHITECTURE.md C3, operations 5-7).

    `simulator` must be the one that generated the world these results were produced in
    — `PairedRun.simulator` — since the Hidden Truths are keyed by billing event and
    scoring against a different world's answer key would silently compare a run to
    somebody else's exam.
    """
    if not results:
        raise ValueError(
            "Cannot score an empty batch against the hidden truth: there is nothing to "
            "compare and no denominator for recall."
        )

    agent = agent_for(results[0])
    mixed = {agent_for(result) for result in results} - {agent}
    if mixed:
        raise ValueError(
            f"against_truth() expects one agent's batch; got {agent!r} mixed with "
            f"{sorted(mixed)!r}."
        )

    frame = truth_frame(results, simulator) if frame is None else frame
    scored = frame[frame["has_truth"]]

    recoverable = scored[scored["is_recoverable"].astype(bool)]
    hopeless = scored[~scored["is_recoverable"].astype(bool)]
    missed = recoverable[~recoverable["recovered"].astype(bool)]

    wasted = int(hopeless["attempt_count"].sum())
    total_attempts = int(frame["attempt_count"].sum())

    return AgainstTruth(
        agent=agent,
        transactions=int(len(frame)),
        scored=int(len(scored)),
        recoverable=int(len(recoverable)),
        unrecoverable=int(len(hopeless)),
        recovered_of_recoverable=int(recoverable["recovered"].sum()),
        recall=(
            float(recoverable["recovered"].astype(bool).mean())
            if len(recoverable)
            else None
        ),
        missed_recoverable=int(len(missed)),
        missed_by_terminal_reason=_value_counts(missed["terminal_reason"]),
        recovered_of_unrecoverable=int(hopeless["recovered"].sum()),
        wasted_attempts=wasted,
        mean_wasted_attempts=_optional_mean(hopeless["attempt_count"]),
        max_wasted_attempts=int(hopeless["attempt_count"].max()) if len(hopeless) else None,
        wasted_attempt_share=(wasted / total_attempts) if total_attempts else None,
        classifier=(
            _classifier_diagnosis(scored) if _consults_classifier(agent) else None
        ),
    )


# ---------------------------------------------------------------------------
# One citable result per run (ARCHITECTURE.md C3, operation 8)
# ---------------------------------------------------------------------------
#
# "Package all of the above into a single exportable results object per batch run, keyed
# by run/seed, so a given comparison is always reproducible and citable."
#
# Both halves of the word matter, and they are different requirements:
#
# *Citable* means a number can be traced back to the exact world it came from. Every
# figure this project has quoted so far has carried the caveat "one seed, one config"
# in prose, in a notes file, next to the table. Prose caveats get separated from their
# tables the moment somebody copies one into a slide. So the configuration travels
# *inside* the results object and into every export it produces.
#
# *Reproducible* means somebody else can get the same numbers back. The seed alone is
# not enough to promise that — the same seed at a different failure rate is a different
# world — so the key is a fingerprint of everything that determined the outcome, and
# `run_evaluation` can rebuild the whole run from that configuration alone.


RESULTS_SCHEMA_VERSION = 1
"""Stamped into every export.

An exported result outlives the code that wrote it: a JSON file sitting in a judge's
folder after the demo has no way to know that the meaning of a field changed afterwards.
Bumping this whenever a field's *meaning* changes (not merely when one is added) is what
lets a future reader tell "this run had zero closures" from "this run predates closures
being counted".
"""


def _jsonable(value: Any) -> Any:
    """Convert a configuration value into something JSON can hold, losslessly enough to
    be read back by a human.

    The configurations carry a few Python-only shapes: `datetime` start times, enum
    channels, and — in `SimulatorConfig.mandate_profile_mix` — a dict keyed by *tuples*
    of channels, which JSON cannot express as a key at all. Tuple keys become
    `"card+upi"`, which is readable and unambiguous, rather than being dropped.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {_config_key(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _config_key(key: Any) -> str:
    """A dict key JSON can hold. A tuple of channels becomes `"card+upi"`."""
    if isinstance(key, tuple):
        return "+".join(str(_jsonable(part)) for part in key)
    return str(_jsonable(key))


def config_dict(config: Any) -> dict[str, Any]:
    """One configuration dataclass as a plain, JSON-safe dict."""
    return {name: _jsonable(value) for name, value in asdict(config).items()}


def run_key(
    simulator_config: SimulatorConfig,
    orchestrator_config: OrchestratorConfig,
    baseline_config: BaselineConfig,
) -> str:
    """A short, stable fingerprint of everything that determined a run's outcome.

    Twelve hex characters of a SHA-256 over the three configurations, serialised
    canonically (sorted keys) so that the same configuration always produces the same key
    and any difference at all produces a different one.

    Deliberately **not** Python's `hash()`, which is randomised per process for strings
    and would give a different key for the same run on every invocation — the same class
    of bug that broke seeded reproducibility in Phase 2 (see `Customer.ordered_channels`).
    A key that changes between processes cannot key anything.

    It identifies the *inputs*, never the outputs: two runs with the same key must
    produce the same numbers, and if they ever do not, the key is what makes that
    visible instead of invisible.
    """
    payload = json.dumps(
        {
            "schema": RESULTS_SCHEMA_VERSION,
            "simulator": config_dict(simulator_config),
            "orchestrator": config_dict(orchestrator_config),
            "baseline": config_dict(baseline_config),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class BatchResults:
    """Everything C3 computed for one batch run, plus the exact configuration that
    produced it (ARCHITECTURE.md C3, operation 8).

    The single object Phase 12's dashboard reads, the single thing a judge can be handed,
    and the unit in which this project's numbers should be quoted from here on. Frozen,
    like every record in this project.
    """

    key: str
    """The `run_key` fingerprint. Short enough to read out loud in a demo, specific
    enough that quoting a number with it makes the number checkable."""

    schema_version: int

    simulator_config: dict[str, Any]
    orchestrator_config: dict[str, Any]
    baseline_config: dict[str, Any]
    """The configurations as plain dicts rather than the dataclasses themselves.

    An exported result has to survive being read by something that has never imported
    this project, and a `SimulatorConfig` cannot. Keeping the dict form as the stored
    field means what is exported is exactly what is held — there is no second,
    slightly-different serialisation step that could disagree with the object it came
    from.
    """

    head_to_head: HeadToHead
    smart_vs_truth: AgainstTruth
    baseline_vs_truth: AgainstTruth

    @property
    def seed(self) -> int:
        return int(self.simulator_config["seed"])

    @property
    def label(self) -> str:
        """A one-line citation for a chart caption or a spoken demo: which world these
        numbers came from, in the four terms that actually vary between runs.

        Pure ASCII, following the rule Phase 10 set for anything that might be printed to
        a Windows console during a live demo.
        """
        return (
            f"run {self.key} (seed {self.seed}, "
            f"{self.simulator_config['num_customers']} customers, "
            f"failure rate {self.simulator_config['base_failure_rate']})"
        )

    def to_dict(self) -> dict[str, Any]:
        """The whole result as nested, JSON-safe primitives."""
        return {
            "key": self.key,
            "schema_version": self.schema_version,
            "label": self.label,
            "config": {
                "simulator": self.simulator_config,
                "orchestrator": self.orchestrator_config,
                "baseline": self.baseline_config,
            },
            "head_to_head": self.head_to_head.to_dict(),
            "against_truth": {
                AGENT_SMART: self.smart_vs_truth.to_dict(),
                AGENT_BASELINE: self.baseline_vs_truth.to_dict(),
            },
        }

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """The result as JSON text, written to `path` only if one is given.

        Returning the text and treating the file as optional is the same choice the Audit
        Trail's export made, for the same reason: Phase 12's `st.download_button` needs
        the string in memory, and a dashboard that had to write a temp file on the demo
        machine to offer a download would be worse in every way.
        """
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=False)
        if path is not None:
            write_text(path, text)
        return text

    def to_frame(self) -> pd.DataFrame:
        """The result as a tidy table: one row per metric, per agent.

        Long-format rather than one wide row per run, because the two things anyone
        actually does with this are (a) read one run's numbers down a column and (b)
        stack several runs and compare them — and a long table does both without
        reshaping, while a wide one does neither well. Every row carries the run key, so
        concatenating several runs' frames never loses track of which world a number came
        from.

        Nested counts (`terminal_reasons`, `closures_by_channel`,
        `missed_by_terminal_reason`) are flattened to dotted metric names rather than
        dropped, so nothing in `to_dict` is missing from the table.
        """
        rows: list[dict[str, Any]] = []

        def add(section: str, agent: Optional[str], payload: dict[str, Any]) -> None:
            for metric, value in payload.items():
                if metric == "agent":
                    continue
                if isinstance(value, dict):
                    for sub, sub_value in value.items():
                        rows.append(
                            {
                                "run_key": self.key,
                                "section": section,
                                "agent": agent,
                                "metric": f"{metric}.{sub}",
                                "value": sub_value,
                            }
                        )
                    continue
                rows.append(
                    {
                        "run_key": self.key,
                        "section": section,
                        "agent": agent,
                        "metric": metric,
                        "value": value,
                    }
                )

        add("head_to_head", AGENT_SMART, self.head_to_head.smart.to_dict())
        add("head_to_head", AGENT_BASELINE, self.head_to_head.baseline.to_dict())
        add(
            "comparison",
            None,
            {
                "recovery_rate_gain_points": self.head_to_head.recovery_rate_gain_points,
                "attempts_saved": self.head_to_head.attempts_saved,
                "attempt_reduction": self.head_to_head.attempt_reduction,
            },
        )

        for agent, scored in (
            (AGENT_SMART, self.smart_vs_truth),
            (AGENT_BASELINE, self.baseline_vs_truth),
        ):
            payload = scored.to_dict()
            classifier = payload.pop("classifier")
            add("against_truth", agent, payload)
            if classifier is not None:
                add("classifier", agent, classifier)

        add("config", None, {f"simulator.{k}": v for k, v in self.simulator_config.items()})
        add("config", None, {f"orchestrator.{k}": v for k, v in self.orchestrator_config.items()})
        add("config", None, {f"baseline.{k}": v for k, v in self.baseline_config.items()})

        return pd.DataFrame(rows, columns=["run_key", "section", "agent", "metric", "value"])

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        """The tidy table as CSV text, written to `path` only if one is given."""
        text = self.to_frame().to_csv(index=False)
        if path is not None:
            write_text(path, text)
        return text


def evaluate(paired: PairedRun) -> BatchResults:
    """Score a finished paired run and package it (C3, operations 1-8).

    The one function the Dashboard and any future caller should use — `head_to_head` and
    `against_truth` remain available for reading a single piece in isolation, but this is
    what produces a result anybody should quote, because it is the only one that stamps
    the numbers with the world they came from.
    """
    return BatchResults(
        key=run_key(
            paired.simulator_config, paired.orchestrator_config, paired.baseline_config
        ),
        schema_version=RESULTS_SCHEMA_VERSION,
        simulator_config=config_dict(paired.simulator_config),
        orchestrator_config=config_dict(paired.orchestrator_config),
        baseline_config=config_dict(paired.baseline_config),
        head_to_head=head_to_head(paired),
        smart_vs_truth=against_truth(paired.smart, paired.simulator),
        baseline_vs_truth=against_truth(paired.baseline, paired.simulator),
    )


def run_evaluation(
    simulator_config: SimulatorConfig,
    orchestrator_config: Optional[OrchestratorConfig] = None,
    baseline_config: Optional[BaselineConfig] = None,
) -> BatchResults:
    """Build the world, run both agents over it, score them, and return the result.

    The whole pipeline in one call, and the operational meaning of "reproducible": hand
    this function the configuration recorded inside any exported result and it rebuilds
    that run from scratch, key included. If the key comes back different, the
    configuration and the numbers have parted company — which is exactly what a
    reproducibility claim needs to be able to detect.
    """
    return evaluate(
        run_paired_batch(
            simulator_config,
            orchestrator_config=orchestrator_config,
            baseline_config=baseline_config,
        )
    )
