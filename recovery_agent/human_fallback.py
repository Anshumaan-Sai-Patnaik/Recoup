"""B5 — Human Fallback / Nudge.

The final arm. Only reached when the Circuit Breaker (B4) has closed *every*
registered channel for a transaction — never on a count of its own, and never
chosen by the Bandit (B2) as an ordinary arm while automated options remain.
Reaching this component ends the automated loop for that transaction: what
follows is a single human-facing message, not another retry (see
notes/ARCHITECTURE.md Part B5 for the full role description).

Plain Python string templates only, per DESIGN.md's B5 row — no templating
engine for the core build's single fixed message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from recovery_agent.models import (
    MandateChannel,
    Merchant,
    TransactionState,
    TransactionStatus,
)

TERMINAL_STATUSES: frozenset[TransactionStatus] = frozenset(
    {
        TransactionStatus.RECOVERED,
        TransactionStatus.ABANDONED,
        TransactionStatus.ESCALATED_TO_HUMAN,
    }
)
"""The statuses that end a transaction's recovery journey. `IN_PROGRESS` is the
only non-terminal one — every other status means no further attempt may ever be
scheduled for this transaction, whether it ended well (recovered) or not
(abandoned / escalated)."""


class TransactionAlreadyTerminalError(RuntimeError):
    """Raised when escalation is requested for a transaction that has already
    finished its journey for some *other* reason (recovered or abandoned).

    Deliberately an error rather than a silent no-op, for the same reason the
    Classifier (B1) raises on an unrecognized decline code: a transaction that
    was already recovered cannot also be escalated to a human, so a caller
    asking for that is an Orchestrator (D1) sequencing bug, and silently
    overwriting the status would hide it *and* corrupt the very outcome the
    Metrics engine (C3) later counts.
    """

    def __init__(self, transaction: TransactionState) -> None:
        self.transaction_id = transaction.transaction_id
        self.status = transaction.status
        super().__init__(
            f"Transaction {transaction.transaction_id!r} cannot be escalated to a "
            f"human: it is already terminal with status {transaction.status.value!r}."
        )


def is_terminal(transaction: TransactionState) -> bool:
    """Whether this transaction's journey has ended and no further attempt may be
    scheduled for it.

    This is the guard the Orchestrator's (D1) per-attempt loop checks before
    starting another round — "stop scheduling further attempts" is enforced by
    the terminal status itself, not by a separate flag that could drift out of
    sync with it.
    """
    return transaction.status in TERMINAL_STATUSES


def escalate_to_human(transaction: TransactionState) -> TransactionState:
    """Mark `transaction` as escalated to a human and end its automated loop
    (ARCHITECTURE.md B5, operation 1).

    Mutates and returns the same `TransactionState` the rest of the system is
    already holding — Transaction State is the one piece of live state every
    component shares (A1, operation 2), so handing back a copy would leave the
    Orchestrator looking at a stale status.

    Idempotent for a transaction that is *already* escalated (re-escalation is a
    harmless no-op), but raises `TransactionAlreadyTerminalError` for one that
    ended any other way — see that exception's own docstring.
    """
    if transaction.status == TransactionStatus.ESCALATED_TO_HUMAN:
        return transaction
    if is_terminal(transaction):
        raise TransactionAlreadyTerminalError(transaction)
    transaction.status = TransactionStatus.ESCALATED_TO_HUMAN
    return transaction


NUDGE_MESSAGE_TEMPLATE = (
    "Hi, we weren't able to process your subscription renewal for {merchant_name}, "
    "and we've stopped trying automatically after {attempt_phrase}. "
    "To keep your subscription active, please update your payment method or complete "
    "the payment yourself."
)
"""The single fixed customer-facing message for the core build (ARCHITECTURE.md B5,
operation 2; DESIGN.md's B5 row keeps this an f-string template rather than pulling in
Jinja2, which only earns its place once there are multiple wording variants to select
between).

Two deliberate choices about what this message does *not* say:

- **It never states why the payment failed.** The system knows the decline category
  internally, but a decline code is the issuer's private reason, and telling a customer
  "your bank said insufficient funds" is both presumptuous and often wrong by the time
  they read it. The message asks for an action instead of diagnosing them — the
  diagnosis belongs in the Audit Trail (C1), which is read by the merchant, not by the
  customer.
- **It is gain-framed, not loss-framed** ("to keep your subscription active" rather
  than "your subscription will be cancelled"). This is the neutral, non-manipulative
  baseline on purpose: if the nudge-theory A/B experiment (IDEA.md §8e) is later built,
  this wording becomes the control arm that loss-framed variants get measured against,
  so it should not already be a persuasion attempt itself.
"""


def _attempt_phrase(attempt_count: int) -> str:
    """The "after N attempts" fragment of the nudge, pluralized — and worded honestly
    when there were no automated attempts at all (a transaction can be escalated
    straight away if its only registered channel hard-declined on the original billing
    charge, which is a genuinely different story to tell the customer than "we tried
    and failed")."""
    if attempt_count <= 0:
        return "checking the payment methods you have on file"
    if attempt_count == 1:
        return "1 automatic retry"
    return f"{attempt_count} automatic retries"


def build_nudge_message(
    transaction: TransactionState, merchant: Optional[Merchant] = None
) -> str:
    """Render the human-facing nudge for an escalated transaction
    (ARCHITECTURE.md B5, operation 2).

    `merchant` is optional so this stays callable from a test or a manual check that
    only has a `TransactionState` in hand; when it isn't supplied the message falls
    back to the transaction's own `merchant_id`. The Orchestrator (D1) always has the
    real `Merchant` and should pass it, so the customer sees "Streamly" rather than
    "merchant_3".

    Generating the message does not escalate anything — `escalate_to_human` owns that
    state change, and this function is pure. Keeping them separate means the message
    can be re-rendered later (e.g. by the Dashboard, C4) without any risk of a
    re-render mutating a transaction's status.
    """
    merchant_name = merchant.display_name if merchant is not None else transaction.merchant_id
    return NUDGE_MESSAGE_TEMPLATE.format(
        merchant_name=merchant_name,
        attempt_phrase=_attempt_phrase(len(transaction.attempts)),
    )


HUMAN_FALLBACK_EVENT_TYPE = "human_fallback"
"""The `event_type` tag every event this component emits carries. The Audit Trail (C1)
will eventually collect raw events from several components and needs to know which
renderer to apply to each — tagging at the source is cheaper and less fragile than
having C1 guess from an event's shape."""


@dataclass(frozen=True)
class HumanFallbackEvent:
    """The raw terminal event recording that a transaction left the automated loop
    (ARCHITECTURE.md B5, operation 3).

    Emitted here in Phase 7 even though the Audit Trail (C1) that consumes it isn't
    built until Phase 10 — deliberately, so that wiring C1 up later is purely additive
    and this component never has to be revisited (PLAN.md's Phase 7 task 3 asks for
    exactly this).

    It carries **facts, not prose**. Rendering the plain-English sentence a judge reads
    ("...because all three of this customer's registered channels were closed") is C1's
    job, and it can only do that honestly if the specific facts that drove the decision
    travel with the event. Hence `closed_channels` and `attempt_count`: they are the
    "because" of this escalation, kept as data so the eventual sentence is generated
    from the real state rather than asserted.

    Frozen because an audit record that can be edited after the fact isn't an audit
    record.
    """

    transaction_id: str
    customer_id: str
    merchant_id: str
    occurred_at: datetime
    attempt_count: int
    message: str
    closed_channels: tuple[MandateChannel, ...] = field(default_factory=tuple)
    event_type: str = HUMAN_FALLBACK_EVENT_TYPE

    def to_dict(self) -> dict[str, Any]:
        """A flat, JSON/CSV-friendly form of this event.

        The Audit Trail's export requirement (ARCHITECTURE.md C1, operation 4) runs
        through `pandas`, which wants flat rows of primitives — so the channel enums
        become plain strings and the timestamp becomes ISO text here, at the one place
        that knows this event's shape, rather than in export code that would have to
        special-case every event type.
        """
        return {
            "event_type": self.event_type,
            "transaction_id": self.transaction_id,
            "customer_id": self.customer_id,
            "merchant_id": self.merchant_id,
            "occurred_at": self.occurred_at.isoformat(),
            "attempt_count": self.attempt_count,
            "closed_channels": ",".join(c.value for c in self.closed_channels),
            "message": self.message,
        }


def run_human_fallback(
    transaction: TransactionState,
    now: datetime,
    merchant: Optional[Merchant] = None,
    closed_channels: Optional[Sequence[MandateChannel]] = None,
) -> HumanFallbackEvent:
    """Run the whole of B5 for one transaction: mark it escalated, render its nudge,
    and emit the terminal event (ARCHITECTURE.md B5, operations 1-3).

    This is the single call the Orchestrator (D1, step 4) makes once the Circuit
    Breaker reports every registered channel closed. The three pieces stay available
    separately (`escalate_to_human`, `build_nudge_message`, `HumanFallbackEvent`) for
    tests and for the Dashboard, but the Orchestrator should never have to remember to
    call three things in the right order for one conceptual act.

    `now` is *simulated* time, taken from the Simulator's clock (A2, operation 7) — the
    Audit Trail timestamps entries in simulated time, so reaching for the real
    wall-clock here would put an event in the log at a time that never happened in the
    run being audited.

    `closed_channels` is the "because" behind this escalation and should be the
    channels the Circuit Breaker (B4) actually reported closed. It is optional only so
    a manual check can skip it; an event emitted without it is still honest (it simply
    records no channels) rather than silently claiming a reason it wasn't given.
    """
    escalate_to_human(transaction)
    return HumanFallbackEvent(
        transaction_id=transaction.transaction_id,
        customer_id=transaction.customer_id,
        merchant_id=transaction.merchant_id,
        occurred_at=now,
        attempt_count=len(transaction.attempts),
        message=build_nudge_message(transaction, merchant),
        closed_channels=tuple(closed_channels or ()),
    )
