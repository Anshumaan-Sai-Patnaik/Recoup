"""C1 — Audit Trail / Explainability Logger (collection layer).

Records a "because" for every decision the agents make, and makes a whole run
inspectable afterwards rather than only watchable once. ARCHITECTURE.md Part C1 calls
this out as answering the explainability grading criterion directly — not optional
polish — and IDEA.md §10 is the reason: every money-related action needs a specific,
human-readable reason attached to it.

This file is built in the order C1's five operations are written, and currently covers
operations 1-4: **collect** the structured decision events the decision components
already emit and hold them as one ordered, queryable trail (operation 1); **render**
each one into the plain-English sentence IDEA.md §10 asks for (operation 2); serve
that back per transaction and across a whole batch, timestamped in simulated time
(operation 3); and **export** the whole log, or any slice of it, to CSV/JSON via
`pandas` (operation 4) — built on top of exactly the same rows, changing none of them.

Collection and rendering are kept strictly apart: the trail stores the original event
objects and the English is generated from them on demand, so a sentence can never drift
from the facts it describes. There is one copy of what happened, and prose is a view of
it.

**Nothing here decides anything, and nothing here interprets anything.** Every component
from Phases 3-9 was already written to emit facts rather than prose, precisely so that
C1 could be added later without any of them being revisited:

    orchestrator.AttemptDecision            one Smart Agent round (B1-B4 + the outcome)
    baseline_agent.BaselineAttemptDecision  one Baseline round (channel, wait, count)
    human_fallback.HumanFallbackEvent       a transaction leaving the automated loop

That is C1's honesty rule (operation 5) made structural: a renderer can only write a
sentence out of facts that actually travelled with the event, so it cannot assert a
"because" the system never computed. The one place this file synthesises anything at
all is `TransactionOutcomeEvent`, and its docstring explains exactly which
already-recorded facts it copies and why it is not an exception to that rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

import pandas as pd

from recovery_agent.baseline_agent import (
    BaselineAttemptDecision,
    BaselineTransactionResult,
)
from recovery_agent import bandit, pacing
from recovery_agent.human_fallback import HumanFallbackEvent
from recovery_agent.models import (
    AttemptOutcome,
    DeclineCategory,
    DeclineCodeRegistry,
    MandateChannel,
    TransactionState,
    TransactionStatus,
)
from recovery_agent.orchestrator import AttemptDecision, TransactionResult

AGENT_SMART = "smart"
AGENT_BASELINE = "baseline"
"""Which agent produced an entry.

The raw events themselves don't carry this, and shouldn't have to: an `AttemptDecision`
describes a round of the loop, not who was running the loop. But the whole point of the
exercise is holding both runs at once (ARCHITECTURE.md C2, operation 5), so the trail
adds it as *provenance* at collection time — the one fact that is genuinely the
collector's to know rather than the event's.

Transaction ids already encode the same thing via their prefixes (`"txn_..."` against
`"baseline_..."`), but those prefixes are configurable, so filtering a batch log by
string prefix would quietly break the moment someone renamed one. This is the label to
filter on.
"""

TRANSACTION_OUTCOME_EVENT_TYPE = "transaction_outcome"


@dataclass(frozen=True)
class TransactionOutcomeEvent:
    """The closing line of one transaction's log: how its journey ended.

    The only event type C1 creates rather than receives, and it is worth being precise
    about why that does not violate the honesty rule this component exists to enforce.
    It **copies** three facts that were already recorded elsewhere — the final
    `TransactionStatus` the agent set, the short machine-readable `terminal_reason` tag
    the agent chose, and how many attempts the transaction made — and asserts nothing of
    its own. No category is guessed and no cause is inferred.

    It exists because without it a transaction that ended `ABANDONED` (ran out of time,
    or hit the belt-and-braces attempt cap) produces no terminal event at all: the
    Human Fallback event fires only when a journey ends with a human being asked. A log
    a judge reads end to end shouldn't simply stop mid-sentence on those, so every
    transaction gets a closing line and the reason tag says which ending it was.

    Frozen, like every other event in this project: a record that can be edited after
    the fact isn't a record.
    """

    transaction_id: str
    customer_id: str
    merchant_id: str
    status: TransactionStatus
    terminal_reason: str
    attempt_count: int
    occurred_at: Optional[datetime] = None
    """When the journey ended, in *simulated* time — derived from the last thing that
    actually happened (see `_terminal_time`).

    Optional, and left genuinely empty rather than filled with a plausible substitute,
    for a transaction that produced no attempt and no fallback event. Every timestamp in
    this project is simulated time from a `SimulatedClock`; reaching for the wall clock,
    or for the billing event's scheduled time as if an attempt had happened then, would
    put a moment into the audit log that never occurred in the run being audited.
    """

    event_type: str = TRANSACTION_OUTCOME_EVENT_TYPE

    def to_dict(self) -> dict[str, Any]:
        """A flat, JSON/CSV-friendly row, in the same shape every other event's
        `to_dict` already returns, so the export needs no per-type special-casing."""
        return {
            "event_type": self.event_type,
            "transaction_id": self.transaction_id,
            "customer_id": self.customer_id,
            "merchant_id": self.merchant_id,
            "status": self.status.value,
            "terminal_reason": self.terminal_reason,
            "attempt_count": self.attempt_count,
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
        }


AuditEvent = Union[
    AttemptDecision,
    BaselineAttemptDecision,
    HumanFallbackEvent,
    TransactionOutcomeEvent,
]
"""Every kind of raw event this trail can hold.

A closed union rather than a structural `Protocol`, deliberately: C1 operation 2 renders
each type into a *different* sentence, so a new event type has to be a considered
addition here (and, next, in the renderer) rather than something that silently lands in
the log with no sentence attached to it.
"""

AgentResult = Union[TransactionResult, BaselineTransactionResult]


def _event_timestamp(event: AuditEvent) -> Optional[datetime]:
    """When an event happened, in simulated time.

    Each event type names its own moment differently, and this is the single place that
    knows the mapping. For a per-attempt decision that moment is `decided_at` — when the
    agent chose — rather than `attempted_at`, because the decision is what is being
    logged and the wait it chose is one of the facts inside it. The attempt's own time
    travels in the record either way.
    """
    for attribute in ("decided_at", "occurred_at"):
        moment = getattr(event, attribute, None)
        if isinstance(moment, datetime):
            return moment
    return None


def _terminal_time(result: AgentResult) -> Optional[datetime]:
    """The last moment that actually happened in a transaction's journey.

    Preference order, most authoritative first: the human-fallback event's own timestamp
    (the journey ended exactly there), then the last attempt's time, then the last
    decision's. Returns `None` for the — in practice unreachable, but not assumed
    impossible — case of a transaction that terminated having done nothing at all, so
    that its outcome event stays honestly untimed instead of inventing a moment.
    """
    if result.human_fallback_event is not None:
        return result.human_fallback_event.occurred_at
    if result.transaction.attempts:
        return result.transaction.attempts[-1].attempted_at
    if result.decisions:
        return _event_timestamp(result.decisions[-1])
    return None


def build_outcome_event(result: AgentResult) -> TransactionOutcomeEvent:
    """Derive the closing event for a finished transaction out of what it already
    recorded — see `TransactionOutcomeEvent` for why this is a copy, not a claim."""
    transaction: TransactionState = result.transaction
    return TransactionOutcomeEvent(
        transaction_id=transaction.transaction_id,
        customer_id=transaction.customer_id,
        merchant_id=transaction.merchant_id,
        status=transaction.status,
        terminal_reason=result.terminal_reason,
        attempt_count=len(transaction.attempts),
        occurred_at=_terminal_time(result),
    )


@dataclass(frozen=True)
class AuditEntry:
    """One event in the trail, plus the provenance the event itself doesn't carry.

    The raw event is kept whole rather than flattened on the way in. That matters twice
    over: the renderer (operation 2) needs the typed facts to write an honest sentence,
    and keeping the original object means the trail can never drift from what the agent
    actually emitted — there is no second, lossy copy of the truth living here.
    """

    sequence: int
    """Position in the run, assigned on arrival. This is the trail's stable tie-breaker:
    several events can share one simulated timestamp (a transaction that escalates does
    so at the same instant as its last attempt), and a log that reordered them between
    two exports of the same run would not be much of an audit record."""

    agent: str
    transaction_id: str
    event_type: str
    occurred_at: Optional[datetime]
    event: AuditEvent

    @property
    def sentence(self) -> str:
        """This entry as one plain-English line (C1, operation 2).

        Rendered on demand rather than stored, so the sentence can never drift from the
        facts it describes: there is exactly one copy of what happened, and the English
        is a view of it.
        """
        return render(self.event, self.agent)

    def to_dict(self) -> dict[str, Any]:
        """A flat row: the collector's provenance columns, the rendered sentence, then
        the event's own facts.

        The event's `to_dict` wins on any shared key — it is the source of truth for its
        own facts, and this wrapper only ever adds context around them.

        The sentence leads the row on purpose. An exported audit trail is read by people
        first and parsed by machines second, so the English belongs where a spreadsheet
        opens on it, with the numbers that justify it in the columns alongside — a reader
        who doubts a sentence can check it against the facts in the same row.
        """
        row: dict[str, Any] = {
            "sequence": self.sequence,
            "agent": self.agent,
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
            "sentence": self.sentence,
        }
        row.update(self.event.to_dict())
        return row


# ---------------------------------------------------------------------------
# Rendering (ARCHITECTURE.md C1, operation 2)
#
# Turning the collected facts into the plain-English sentence IDEA.md §10 asks for:
#
#     "Retrying in 6 hours via UPI instead of card, because this card has had 3 soft
#      declines in the past 24h..."
#
# One rule governs every function below, and it is the one C1 operation 5 exists for:
# **every clause of a "because" must be traceable to a field on the event being
# rendered.** No sentence here reaches for a plausible-sounding cause the system never
# computed. Where a fact is genuinely absent — the baseline has no category, the pacing
# dial hasn't seen enough attempts to judge system health — the sentence says so out
# loud instead of quietly omitting it or filling it in.
#
# The four event types get four genuinely different sentences rather than one template
# with blanks, because the agents did genuinely different things and describing the
# naive one in the smart one's vocabulary would flatter it.
#
# Every rendered sentence is **pure ASCII**, for the same reason B5's nudge message is
# (see `human_fallback.NUDGE_MESSAGE_TEMPLATE`): a live demo that prints a line to a
# Windows console inherits a cp1252 encoding that raises on an em dash. The exported
# files are UTF-8 and would carry one happily, but the console is where this log gets
# read out loud, and a `UnicodeEncodeError` mid-demo is a bad way to find that out.
# ---------------------------------------------------------------------------

_CHANNEL_NAMES: dict[MandateChannel, str] = {
    MandateChannel.CARD: "card",
    MandateChannel.UPI: "UPI",
    MandateChannel.NETBANKING: "netbanking",
}

_CATEGORY_PHRASES: dict[DeclineCategory, str] = {
    DeclineCategory.SOFT: "soft (temporary)",
    DeclineCategory.HARD: "hard (permanent)",
}

_STATUS_PHRASES: dict[TransactionStatus, str] = {
    TransactionStatus.RECOVERED: "recovered",
    TransactionStatus.ESCALATED_TO_HUMAN: "escalated to a human",
    TransactionStatus.ABANDONED: "abandoned",
    TransactionStatus.IN_PROGRESS: "still in progress",
}

_REASON_PHRASES: dict[str, str] = {
    "recovered": "the payment went through",
    "all_channels_closed": "every channel this customer had on file was permanently "
    "closed, so no automated option was left",
    "recovery_horizon_exceeded": "the recovery window for this billing cycle ran out "
    "before another attempt could be made",
    "max_attempts_reached": "the overall per-transaction attempt ceiling was reached",
    "unrecognized_decline_code": "a decline code came back that is not in the registry, "
    "so no further automated decision could be made honestly",
    "attempt_cap_reached": "its fixed attempt cap was spent",
}
"""English for the machine-readable `terminal_reason` tags both agents set.

A tag with no entry here renders as the raw tag rather than as a guess — a reason
invented by the log because a lookup missed would be exactly the failure this component
is supposed to prevent.
"""


def _channel(channel: MandateChannel) -> str:
    return _CHANNEL_NAMES.get(channel, channel.value)


def _channel_list(channels: Iterable[MandateChannel]) -> str:
    """A readable "card, UPI and netbanking" out of a channel sequence."""
    names = [_channel(c) for c in channels]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _duration(hours: float) -> str:
    """A wait, in the unit a person would actually say it in."""
    if hours < 1:
        minutes = int(round(hours * 60))
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    if hours < 48:
        return f"{hours:.1f} hours"
    return f"{hours / 24:.1f} days"


def _percent(rate: float) -> str:
    return f"{rate * 100:.0f}%"


def _category(category: Optional[DeclineCategory]) -> str:
    if category is None:
        return "uncategorised"
    return _CATEGORY_PHRASES.get(category, category.value)


def _code_phrase(channel: MandateChannel, code: str) -> str:
    """`"51 (Insufficient funds)"` when the registry knows the code, `"51"` when it
    doesn't — the description is a lookup in reference data, never an interpretation.

    The registry's wording is quoted verbatim rather than case-folded to fit the
    sentence: several descriptions carry acronyms ("UPI mandate revoked", "PSP timeout")
    that lower-casing would mangle, and a log that silently rewrites the reference data
    it is citing is a worse trade than a capital letter mid-sentence."""
    description = DeclineCodeRegistry.describe(channel, code)
    return f"{code} ({description})" if description else code


def _attempts_phrase(count: int) -> str:
    if count == 0:
        return "no automatic retries"
    if count == 1:
        return "1 automatic retry"
    return f"{count} automatic retries"


def _render_attempt_decision(event: AttemptDecision) -> str:
    """The Smart Agent's round, in English — the sentence IDEA.md §10 is describing.

    Every clause below is read off the event: which channels the Circuit Breaker
    reported open, which arms the Bandit was allowed to choose from, what the failure it
    was reacting to was, what Pacing's dial and health signal said, and whether a safety
    rule moved the attempt. Nothing is inferred.

    The one derived fact is which channels are *permanently* closed as against merely
    waiting out a spacing window: a channel that is `"closed"` in the Circuit Breaker's
    status but still present in the Bandit's available arms is serving out a wait, not
    dead (see `bandit.available_arms`). That distinction is what lets the log say "card
    is closed for now" rather than the much stronger, and often wrong, "card is dead".
    """
    chosen = _channel(event.chosen_channel)
    switched = event.chosen_channel != event.context_channel
    action = (
        f"Attempt {event.attempt_number} on {event.transaction_id}: retrying via "
        f"{chosen}"
        + (f" instead of {_channel(event.context_channel)}" if switched else "")
        + f", {_duration(event.actual_wait_hours)} after the last decline"
    )

    arms = set(event.available_arms)
    dead = [c for c, status in event.channel_status if c not in arms]
    waiting = [
        c for c, status in event.channel_status if status == "closed" and c in arms
    ]

    reasons: list[str] = []
    if dead:
        reasons.append(
            f"the circuit breaker has permanently closed {_channel_list(dead)}"
        )
    if len(event.available_arms) == 1:
        reasons.append(f"{chosen} is the only channel it is still allowed to try")
    elif event.selection_mode == bandit.SELECTION_EXPLORE:
        # Epsilon-greedy explores a fixed fraction of the time, picking uniformly at
        # random rather than by record. Describing that as "going on what has worked
        # before" would credit the choice to evidence it did not use — the exact class
        # of unconfirmed claim C1 operation 5 forbids. The event carries which half ran
        # (`selection_mode`) precisely so this sentence doesn't have to assume.
        reasons.append(
            f"the bandit picked it at random from the {len(event.available_arms)} "
            f"channels still allowed ({_channel_list(event.available_arms)}) as a "
            "deliberate exploration step, so it keeps gathering evidence on options it "
            "has tried less often"
        )
    else:
        # Stated as the rule that was applied rather than as a claim about the
        # evidence, because this renderer cannot see the Bandit's pool: at cold start
        # every arm scores 0.0 and the tie-break decides. Naming the tie-break makes
        # the sentence true in that case too, instead of implying a track record that
        # may not exist yet.
        reasons.append(
            f"the bandit chose it from the {len(event.available_arms)} channels still "
            f"allowed ({_channel_list(event.available_arms)}) by its usual rule: the "
            "best success rate on record after a "
            f"{_category(event.context_category)} decline on "
            f"{_channel(event.context_channel)}, ties going to the earliest-registered "
            "channel"
        )
    # A channel the chosen one is *not* gets a plain "closed for now" note. The chosen
    # channel being closed at decision time is a different story and is told by the
    # spacing clause below, or — when Pacing's own wait already outlasts the closure —
    # by the "reopens by then" clause, which is a fact of the run rather than a
    # prediction: the Orchestrator re-checks the channel at the moment it fires and
    # refuses to attempt a closed one, so this attempt happening means it had reopened.
    others_waiting = [c for c in waiting if c != event.chosen_channel]
    if others_waiting:
        reasons.append(
            f"{_channel_list(others_waiting)} is closed for the moment but not written off"
        )
    if event.chosen_channel in waiting and not event.spacing_rule_applied:
        reasons.append(f"{chosen} is closed right now but reopens before the wait is up")

    if event.health_signal == pacing.SIGNAL_WARMING_UP:
        reasons.append(
            "the system has not yet seen enough attempts to judge its own health, so "
            f"pacing is running at its starting setting of {event.aggressiveness:.2f}"
        )
    else:
        health = (
            "system health looks normal"
            if event.health_signal == pacing.SIGNAL_HEALTHY
            else "system health is degraded, so pacing has backed off"
        )
        reasons.append(
            f"{health} ({_percent(event.rolling_success_rate)} of recent attempts "
            f"succeeded against a baseline of "
            f"{_percent(event.baseline_success_rate)}), putting the pacing dial at "
            f"{event.aggressiveness:.2f} and the wait at "
            f"{_duration(event.base_wait_hours)} before jitter"
        )
    if event.spacing_rule_applied:
        # `circuit_breaker.earliest_next_attempt_at` implements exactly one time-based
        # rule — card's 24h soft-decline spacing — so naming it here is a statement of
        # what actually moved the attempt, not a guess at which rule it might have been.
        reasons.append(
            f"the {chosen} channel's 24-hour retry-spacing rule then pushed the attempt "
            f"back from {_duration(event.jittered_wait_hours)} to "
            f"{_duration(event.actual_wait_hours)}"
        )

    return f"{action}, because {'; '.join(reasons)}. {_render_outcome_clause(event)}"


def _render_outcome_clause(event: AttemptDecision) -> str:
    """What the bank said back, and what — if anything — the Classifier made of it."""
    if event.outcome == AttemptOutcome.APPROVED:
        return "The attempt was approved and the payment recovered."
    if event.decline_code is None:
        return "The attempt was declined, with no code returned."
    if event.unrecognized_decline_code or event.decline_category is None:
        return (
            f"The attempt was declined with code {event.decline_code}, which is not in "
            "the decline-code registry, so no soft/hard category is being claimed for it."
        )
    return (
        f"The attempt was declined with code "
        f"{_code_phrase(event.chosen_channel, event.decline_code)}, which the classifier "
        f"reads as a {_category(event.decline_category)} decline."
    )


def _render_baseline_attempt_decision(event: BaselineAttemptDecision) -> str:
    """The Baseline Agent's round, in its own honest English.

    Deliberately not the Smart Agent's sentence with the missing parts left blank. There
    is no channel comparison to describe because no channel was compared, no health
    reading because none was taken, and no diagnosis because there is no classifier. The
    sentence says what the agent actually did — the same thing it does every time — and
    naming the decline code it recorded but never read is the single most useful thing
    this log can show a judge about why the naive approach loses.
    """
    action = (
        f"Attempt {event.attempt_number} of {event.attempt_cap} on "
        f"{event.transaction_id}: retrying via {_channel(event.chosen_channel)}, "
        f"{_duration(event.wait_hours)} after the last decline, because this agent "
        "always retries the same channel on a fixed schedule; it did not look at the "
        "decline code, consider any other payment method on file, or check any safety "
        "rule"
    )
    if event.outcome == AttemptOutcome.APPROVED:
        return f"{action}. The attempt was approved and the payment recovered."
    if event.decline_code is None:
        return f"{action}. The attempt was declined, with no code returned."
    return (
        f"{action}. The attempt was declined with code {event.decline_code}; this agent "
        "records the code but never classifies it, so no soft/hard category exists for "
        "this attempt."
    )


def _render_human_fallback(event: HumanFallbackEvent, agent: str) -> str:
    """A transaction leaving the automated loop.

    The `because` is taken from `closed_channels`, and the empty case is where the
    honesty rule bites hardest: the Baseline Agent's escalations *always* carry an empty
    list, because it never closed anything — it simply ran out of counter. Rendering
    that as "every channel was closed" would hand the naive agent a safety story it has
    no claim to, and it is the single most tempting mistake available to this renderer.
    """
    who = f"Transaction {event.transaction_id} has been handed to a human " + (
        "without a single automatic retry"
        if event.attempt_count == 0
        else f"after {_attempts_phrase(event.attempt_count)}"
    )
    if event.closed_channels:
        because = (
            "because the circuit breaker had permanently closed every channel this "
            f"customer had on file ({_channel_list(event.closed_channels)})"
        )
    elif agent == AGENT_BASELINE:
        because = (
            "because its fixed attempt cap ran out; no channel was ever checked, and "
            "none was closed"
        )
    else:
        because = (
            "with no channel reported closed; the reason the loop stopped is on this "
            "transaction's closing line"
        )
    return f'{who}, {because}. The customer is sent: "{event.message}"'


def _render_transaction_outcome(event: TransactionOutcomeEvent) -> str:
    """The closing line: how the journey ended, in the agent's own recorded terms."""
    status = _STATUS_PHRASES.get(event.status, event.status.value)
    if not event.terminal_reason:
        reason = "no reason tag was recorded"
    else:
        reason = _REASON_PHRASES.get(
            event.terminal_reason, f"recorded reason: {event.terminal_reason}"
        )
    return (
        f"Transaction {event.transaction_id} finished as {status} after "
        f"{event.attempt_count} attempt{'s' if event.attempt_count != 1 else ''}: "
        f"{reason}."
    )


def render(event: AuditEvent, agent: str = AGENT_SMART) -> str:
    """Render any collected event as one plain-English sentence
    (ARCHITECTURE.md C1, operation 2).

    `agent` only affects the human-fallback sentence, where the same event shape means
    two different things depending on who produced it — see `_render_human_fallback`.

    An event type with no renderer raises rather than falling back to a generic line: a
    log entry that says nothing in particular is worse than a missing one, because it
    looks like a record.
    """
    if isinstance(event, AttemptDecision):
        return _render_attempt_decision(event)
    if isinstance(event, BaselineAttemptDecision):
        return _render_baseline_attempt_decision(event)
    if isinstance(event, HumanFallbackEvent):
        return _render_human_fallback(event, agent)
    if isinstance(event, TransactionOutcomeEvent):
        return _render_transaction_outcome(event)
    raise TypeError(f"No audit-trail renderer for {type(event).__name__!r}.")


def agent_for(result: AgentResult) -> str:
    """Which agent produced a result, read off its type.

    Inferred rather than passed in wherever possible, because the alternative — a caller
    labelling each batch by hand — is a mislabelling waiting to happen, and a head-to-
    head comparison whose two halves can be swapped by one wrong string argument is not
    a comparison anyone should trust.

    Public rather than private because the Metrics engine (C3) labels the same two runs
    and must label them the same way — a second copy of this mapping living over there
    is exactly the kind of duplicate that drifts.
    """
    if isinstance(result, TransactionResult):
        return AGENT_SMART
    if isinstance(result, BaselineTransactionResult):
        return AGENT_BASELINE
    raise TypeError(
        f"Cannot record an audit trail for {type(result).__name__!r}: expected an "
        "orchestrator.TransactionResult or a baseline_agent.BaselineTransactionResult."
    )


# ---------------------------------------------------------------------------
# Export (ARCHITECTURE.md C1, operation 4)
#
# "Support export of the full log (or a filtered slice) in a structured, downloadable
# form — not just something visible transiently in a live view." IDEA.md §10 gives the
# reason: the live narration is seen once, by whoever is in the room. A judge reviewing
# the repo afterwards, or anyone who wants to check a claim we made on stage, needs the
# same log as a file they can open, sort and grep on their own.
#
# There is no new rendering and no new fact here. An export is just `AuditEntry.to_dict`
# run over a list of entries and handed to `pandas` — which is the whole reason every
# event type was written to flatten itself back in Phases 3-9.
# ---------------------------------------------------------------------------

EXPORT_COLUMN_ORDER: tuple[str, ...] = (
    # provenance the trail added
    "sequence",
    "agent",
    "occurred_at",
    "sentence",
    # which record this is, and whose
    "event_type",
    "transaction_id",
    "customer_id",
    "merchant_id",
    # where in the journey it sits
    "attempt_number",
    "attempt_cap",
    "attempt_count",
    # what the agent was reacting to
    "context_channel",
    "context_category",
    "channel_status",
    "closed_channels",
    # what it chose
    "available_arms",
    "chosen_channel",
    "chosen_arm",
    "selection_mode",
    # why it waited as long as it did
    "rolling_success_rate",
    "baseline_success_rate",
    "health_signal",
    "aggressiveness",
    "base_wait_hours",
    "jittered_wait_hours",
    "actual_wait_hours",
    "spacing_rule_applied",
    # what happened
    "decided_at",
    "attempted_at",
    "outcome",
    "decline_code",
    "decline_category",
    "unrecognized_decline_code",
    # how the journey ended
    "status",
    "terminal_reason",
    "message",
)
"""The column order of an exported trail, fixed here rather than left to whatever order
the first row happened to introduce its keys in.

Two reasons it is spelled out. First, **reproducibility**: rows come from four different
event types with overlapping-but-different fields, so a first-seen ordering would shift
depending on which agent or which slice was exported — and an audit artefact whose
columns move between two exports of the same run is a poor audit artefact. Second, it
reads as the decision itself reads: who and when, the English sentence, then what the
agent saw, what it chose, why it waited, what came back, and how the journey ended. A
reader who doubts a sentence can walk left to right through the facts behind it.

A key not listed here is not dropped — it is appended, alphabetically, after these. So
adding a field to any event still lands in the export automatically; listing it here is
only how it gets a considered *position*.
"""


def _ordered_columns(rows: Iterable[dict[str, Any]]) -> list[str]:
    """The columns present in `rows`, in `EXPORT_COLUMN_ORDER`, with anything unlisted
    appended alphabetically.

    Only columns that actually occur are returned, so exporting a slice of one event
    type gives a narrow, readable table rather than a wide one padded with empty
    columns for events it doesn't contain.
    """
    present: set[str] = set()
    for row in rows:
        present.update(row)
    known = [c for c in EXPORT_COLUMN_ORDER if c in present]
    unknown = sorted(present.difference(EXPORT_COLUMN_ORDER))
    return known + unknown


def write_text(path: Union[str, Path], text: str) -> None:
    """Write an export to disk as UTF-8, with newlines left exactly as produced.

    Both details are deliberate. UTF-8 is pinned because a run started from a Windows
    console inherits a cp1252 default that would fail on the first non-ASCII character
    in a merchant name. `newline=""` stops Python translating the line endings `pandas`
    already chose, which would otherwise give a CSV doubled-up `

` line breaks on
    Windows and confuse some spreadsheet readers.

    Public because the Metrics engine (C3) exports files under exactly the same two
    constraints, and a second copy of this convention is one that could drift.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


@dataclass
class AuditTrail:
    """The running log for one run — or for both agents' runs held side by side
    (ARCHITECTURE.md C1, operations 1 and 3).

    Mutable and in-memory, per DESIGN.md's C1 row: events are appended as a batch is
    processed and the trail lives exactly as long as the run it describes. The entries
    inside it are frozen; only the list they sit in grows.

    Insertion order is the trail's canonical order, because it is the order the run
    genuinely produced — which, for a batch processed one transaction at a time, groups
    each journey together and reads naturally end to end. `by_simulated_time` gives the
    other view (what the whole population was doing hour by hour) for when that is the
    question being asked.
    """

    entries: list[AuditEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(self.entries)

    def record(self, event: AuditEvent, agent: str) -> AuditEntry:
        """Append one raw event to the trail (C1, operation 1).

        The single entry point every other collection method funnels through, so that
        sequence numbering and timestamp extraction happen in exactly one place.
        """
        entry = AuditEntry(
            sequence=len(self.entries),
            agent=agent,
            transaction_id=event.transaction_id,
            event_type=event.event_type,
            occurred_at=_event_timestamp(event),
            event=event,
        )
        self.entries.append(entry)
        return entry

    def collect_transaction(
        self, result: AgentResult, agent: Optional[str] = None
    ) -> list[AuditEntry]:
        """Collect everything one finished transaction produced, in the order it
        happened: each round's decision, then the human-fallback event if the journey
        ended that way, then the derived closing line.

        `agent` is inferred from the result's type and should almost never be passed —
        see `agent_for`.
        """
        agent = agent or agent_for(result)
        recorded = [self.record(decision, agent) for decision in result.decisions]
        if result.human_fallback_event is not None:
            recorded.append(self.record(result.human_fallback_event, agent))
        recorded.append(self.record(build_outcome_event(result), agent))
        return recorded

    def collect_batch(
        self, results: Iterable[AgentResult], agent: Optional[str] = None
    ) -> "AuditTrail":
        """Collect a whole agent's run, transaction by transaction. Returns `self`, so a
        paired run can be assembled in one expression."""
        for result in results:
            self.collect_transaction(result, agent)
        return self

    def for_transaction(self, transaction_id: str) -> list[AuditEntry]:
        """That one transaction's running log (C1, operation 3) — the per-transaction
        view the Dashboard (C4) drills into from the batch feed."""
        return [e for e in self.entries if e.transaction_id == transaction_id]

    def for_agent(self, agent: str) -> list[AuditEntry]:
        """One agent's slice of a trail holding both — part of the filtering behind C1
        operation 4's "the full log (or a filtered slice)"."""
        return [e for e in self.entries if e.agent == agent]

    def of_type(self, event_type: str) -> list[AuditEntry]:
        """Every entry of one event type — e.g. just the escalations."""
        return [e for e in self.entries if e.event_type == event_type]

    def transaction_ids(self) -> list[str]:
        """Every transaction the trail has seen, in first-appearance order."""
        seen: dict[str, None] = {}
        for entry in self.entries:
            seen.setdefault(entry.transaction_id, None)
        return list(seen)

    def sentences(self, entries: Optional[Iterable[AuditEntry]] = None) -> list[str]:
        """The trail (or any slice of it) as plain-English lines (C1, operation 2).

        Pass a slice — `trail.sentences(trail.for_agent(AGENT_SMART))` — to narrate part
        of a run; pass nothing for the whole thing.
        """
        return [entry.sentence for entry in (self.entries if entries is None else entries)]

    def narrate(self, transaction_id: str) -> list[str]:
        """One transaction's story, timestamped in simulated time and read top to bottom
        (C1, operation 3).

        This is the per-transaction running log in the form a person actually reads it,
        and the view the Dashboard (C4) drills into from the live batch feed.
        """
        lines: list[str] = []
        for entry in self.for_transaction(transaction_id):
            stamp = entry.occurred_at.isoformat(sep=" ") if entry.occurred_at else "--"
            lines.append(f"[{stamp}] {entry.sentence}")
        return lines

    def by_simulated_time(self) -> list[AuditEntry]:
        """The trail re-ordered as the simulated world experienced it, rather than as
        the batch loop happened to produce it.

        An entry with no timestamp of its own (only an outcome event for a transaction
        that never acted can be one) inherits the last timestamp seen before it, so it
        stays next to the journey it closes instead of being flung to one end of the
        log. `sequence` breaks every tie, so this ordering is stable across exports.
        """
        keyed: list[tuple[datetime, int, AuditEntry]] = []
        last_seen: Optional[datetime] = None
        for entry in self.entries:
            if entry.occurred_at is not None:
                last_seen = entry.occurred_at
            keyed.append((last_seen or datetime.min, entry.sequence, entry))
        keyed.sort(key=lambda item: (item[0], item[1]))
        return [entry for _, _, entry in keyed]

    # -- Export (C1, operation 4) ------------------------------------------------

    def to_rows(
        self, entries: Optional[Iterable[AuditEntry]] = None
    ) -> list[dict[str, Any]]:
        """The trail (or a slice of it) as flat rows of primitives.

        The shared step under every export below, and the seam the Dashboard (C4) can
        also read directly if it wants a table without going through a file.
        """
        return [e.to_dict() for e in (self.entries if entries is None else entries)]

    def to_frame(self, entries: Optional[Iterable[AuditEntry]] = None) -> pd.DataFrame:
        """The trail (or a slice) as a `pandas.DataFrame`, in `EXPORT_COLUMN_ORDER`.

        Built with `dtype=object` on purpose. The four event types carry different
        fields, so most columns are absent from some rows; pandas' normal response is to
        widen an integer column to float to make room for `NaN`, and an audit log
        reporting "attempt 3.0" would be a small, avoidable way of looking untrustworthy.
        Object dtype keeps every value exactly as its event emitted it and leaves the
        gaps genuinely empty — which is the honest rendering anyway: a baseline row has
        no `aggressiveness` because that agent has no pacing dial, not because the value
        is zero or missing.

        An empty trail returns an empty frame with no columns rather than raising —
        exporting a run in which nothing happened is a legitimate, if dull, thing to do.
        """
        rows = self.to_rows(entries)
        return pd.DataFrame(rows, columns=_ordered_columns(rows), dtype=object)

    def to_csv(
        self,
        path: Optional[Union[str, Path]] = None,
        entries: Optional[Iterable[AuditEntry]] = None,
    ) -> str:
        """Export as CSV text, and write it to `path` if one is given.

        The text is returned either way, because the two consumers want different
        things: a saved file for a judge reviewing the repo, and an in-memory string for
        the Dashboard's `st.download_button` (DESIGN.md §3), which hands the browser
        bytes rather than a path on the machine running the demo.

        CSV is the format to open in a spreadsheet — one decision per row, the English
        sentence in the fourth column, and the facts that justify it in the columns
        beside it.
        """
        text = self.to_frame(entries).to_csv(index=False)
        if path is not None:
            write_text(path, text)
        return text

    def to_json(
        self,
        path: Optional[Union[str, Path]] = None,
        entries: Optional[Iterable[AuditEntry]] = None,
        indent: int = 2,
    ) -> str:
        """Export as JSON text (a list of row objects), and write it to `path` if one is
        given.

        `orient="records"` is the shape a person expects — one object per decision,
        keyed by column name — rather than pandas' column-major default, which is
        compact but unreadable. Indented by default for the same reason: this artefact
        exists to be read.

        Missing fields come out as `null`, which is the JSON way of saying what the
        blank CSV cell says: this agent never produced that fact.
        """
        text = self.to_frame(entries).to_json(orient="records", indent=indent)
        if path is not None:
            write_text(path, text)
        return text


def collect_paired_run(
    smart: Iterable[AgentResult], baseline: Iterable[AgentResult]
) -> AuditTrail:
    """Build one trail holding both agents' runs over the same seeded world — the input
    the head-to-head views in Phase 11 (C3) and Phase 12 (C4) read.

    Takes the two result lists rather than a `baseline_agent.PairedRun` so that a trail
    can also be built for one agent alone, and so this module keeps no opinion about how
    the pair was produced.
    """
    trail = AuditTrail()
    trail.collect_batch(smart, AGENT_SMART)
    trail.collect_batch(baseline, AGENT_BASELINE)
    return trail
