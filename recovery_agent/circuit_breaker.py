"""B4 — Circuit Breaker.

The safety net: the hard outer boundary nothing above (Bandit, Pacing) is allowed to
cross. Stateless by design — every fact it needs (attempt counts, timestamps, decline
categories) is read straight off a `TransactionState`'s own `AttemptRecord` list rather
than tracked separately, so there is never a second source of truth to go stale
(DESIGN.md B4; see notes/ARCHITECTURE.md Part B4 for the full role description).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from recovery_agent.models import (
    AttemptRecord,
    DeclineCategory,
    MandateChannel,
    TransactionState,
)

# Visa Excessive Reattempts Rule — a real published rule, not an assumption.
CARD_ROLLING_WINDOW_DAYS = 30
CARD_MAX_ATTEMPTS_PER_WINDOW = 15

# Minimum spacing between soft-decline card retries — also part of the Visa
# Excessive Reattempts Rule guidance (not one of our own assumed caps).
CARD_MIN_RETRY_SPACING_HOURS = 24

# --- ASSUMPTION, not a published rule -------------------------------------------
# Neither UPI Autopay nor netbanking SI has a network-published cap we found
# equivalent to Visa's rule (ARCHITECTURE.md B4, operation 4 explicitly calls for
# "our own conservative, explicitly-labeled-as-assumption caps" here — not a real
# rule). We picked 5 attempts within a rolling 7-day window for both channels: loose
# enough not to prematurely cut off a genuinely recoverable transaction, tight enough
# to still act as a real safety net. Any component surfacing a closure caused by
# these constants (e.g. the Audit Trail) must say so is an assumption, not a rule.
UPI_ROLLING_WINDOW_DAYS = 7
UPI_MAX_ATTEMPTS_PER_WINDOW = 5
NETBANKING_ROLLING_WINDOW_DAYS = 7
NETBANKING_MAX_ATTEMPTS_PER_WINDOW = 5
# ---------------------------------------------------------------------------------

_ASSUMED_CAPS: dict[MandateChannel, tuple[float, int]] = {
    MandateChannel.UPI: (UPI_ROLLING_WINDOW_DAYS, UPI_MAX_ATTEMPTS_PER_WINDOW),
    MandateChannel.NETBANKING: (
        NETBANKING_ROLLING_WINDOW_DAYS,
        NETBANKING_MAX_ATTEMPTS_PER_WINDOW,
    ),
}


def _attempts_in_window(
    transaction: TransactionState,
    channel: MandateChannel,
    now: datetime,
    window_days: Optional[float] = None,
) -> list[AttemptRecord]:
    """This transaction's attempt records for one channel, scoped straight off its
    own `AttemptRecord` list — no separate per-channel counters are tracked anywhere
    (ARCHITECTURE.md B4, operation 1).

    If `window_days` is given, only attempts within that rolling window ending at
    `now` are counted (e.g. the Visa 15-attempts/30-days rule below); otherwise every
    attempt ever recorded for that channel on this transaction is returned.
    """
    attempts = transaction.attempts_for_channel(channel)
    if window_days is None:
        return attempts
    cutoff = now - timedelta(days=window_days)
    return [a for a in attempts if a.attempted_at >= cutoff]


def _channel_attempt_count(
    transaction: TransactionState,
    channel: MandateChannel,
    now: datetime,
    window_days: Optional[float] = None,
) -> int:
    return len(_attempts_in_window(transaction, channel, now, window_days))


def _most_recent_attempt(
    transaction: TransactionState, channel: MandateChannel
) -> Optional[AttemptRecord]:
    """The most recent attempt record on this channel, or None if the channel has
    never been attempted on this transaction yet."""
    attempts = transaction.attempts_for_channel(channel)
    if not attempts:
        return None
    return max(attempts, key=lambda a: a.attempted_at)


def _last_attempt_at(
    transaction: TransactionState, channel: MandateChannel
) -> Optional[datetime]:
    """The timestamp of the most recent attempt on this channel, or None if the
    channel has never been attempted on this transaction yet."""
    attempt = _most_recent_attempt(transaction, channel)
    return attempt.attempted_at if attempt else None


def _card_closed_by_visa_rule(transaction: TransactionState, now: datetime) -> bool:
    """Visa Excessive Reattempts Rule: the card channel closes once this transaction
    has made 15 or more card attempts within the trailing 30 (simulated) days
    (ARCHITECTURE.md B4, operation 2). This is a real published card-network rule,
    not one of our own conservative assumptions — see `CARD_ROLLING_WINDOW_DAYS` /
    `CARD_MAX_ATTEMPTS_PER_WINDOW` above.
    """
    count = _channel_attempt_count(
        transaction, MandateChannel.CARD, now, window_days=CARD_ROLLING_WINDOW_DAYS
    )
    return count >= CARD_MAX_ATTEMPTS_PER_WINDOW


def _card_closed_by_spacing_rule(transaction: TransactionState, now: datetime) -> bool:
    """Minimum ~24h spacing between soft-decline card retries (ARCHITECTURE.md B4,
    operation 3). If the most recent card attempt on this transaction was a soft
    decline and fewer than `CARD_MIN_RETRY_SPACING_HOURS` (simulated) hours have
    elapsed since it, the card channel is temporarily closed — not permanently, just
    not eligible again until the spacing window passes.

    A hard decline doesn't trigger this rule on its own; it closes the channel
    outright via operation 5 (see `_card_closed_by_hard_decline`, task 5) instead.
    """
    last = _most_recent_attempt(transaction, MandateChannel.CARD)
    if last is None or last.decline_category != DeclineCategory.SOFT:
        return False
    elapsed = now - last.attempted_at
    return elapsed < timedelta(hours=CARD_MIN_RETRY_SPACING_HOURS)


def _closed_by_assumed_cap(
    transaction: TransactionState, channel: MandateChannel, now: datetime
) -> bool:
    """UPI/netbanking's own conservative, assumption-labeled attempt caps
    (ARCHITECTURE.md B4, operation 4) — see the `_ASSUMED_CAPS` table above. Not
    applicable to card, which has its own real-rule handling above.
    """
    if channel not in _ASSUMED_CAPS:
        return False
    window_days, max_attempts = _ASSUMED_CAPS[channel]
    count = _channel_attempt_count(transaction, channel, now, window_days=window_days)
    return count >= max_attempts


def _closed_by_hard_decline(transaction: TransactionState, channel: MandateChannel) -> bool:
    """Any channel closes immediately, regardless of attempt count, the moment its
    most recent attempt on this transaction came back hard (ARCHITECTURE.md B4,
    operation 5). A hard decline is a permanent signal (e.g. lost/stolen card, mandate
    revoked) — no rolling window or spacing check applies, unlike the soft-decline
    rules above.
    """
    last = _most_recent_attempt(transaction, channel)
    return last is not None and last.decline_category == DeclineCategory.HARD


def is_channel_open(
    transaction: TransactionState, channel: MandateChannel, now: datetime
) -> bool:
    """Whether `channel` may still be attempted again for this transaction right now
    — the single source of truth combining every closure rule above (Visa rule +
    24h spacing + assumed UPI/netbanking caps + immediate hard-decline closure).
    """
    if _closed_by_hard_decline(transaction, channel):
        return False
    if channel == MandateChannel.CARD:
        if _card_closed_by_visa_rule(transaction, now):
            return False
        if _card_closed_by_spacing_rule(transaction, now):
            return False
        return True
    if _closed_by_assumed_cap(transaction, channel, now):
        return False
    return True


def get_channel_status(
    transaction: TransactionState,
    registered_channels: list[MandateChannel],
    now: datetime,
) -> dict[MandateChannel, str]:
    """The Bandit's (B2) required per-attempt input: for every channel this customer
    has registered, whether it is currently `"open"` (further attempts permitted) or
    `"closed"` (must not be attempted again for this transaction) — ARCHITECTURE.md
    B4, operation 6. Only registered channels are reported; a channel the customer
    never had on file is meaningless to report a status for.
    """
    return {
        channel: "open" if is_channel_open(transaction, channel, now) else "closed"
        for channel in registered_channels
    }


def all_channels_closed(
    transaction: TransactionState,
    registered_channels: list[MandateChannel],
    now: datetime,
) -> bool:
    """The Orchestrator's (D1) trigger condition for routing to Human Fallback (B5):
    every one of this customer's registered channels is closed — never a count-based
    trigger of its own, purely derived from `get_channel_status` (ARCHITECTURE.md B4,
    operation 7).
    """
    status = get_channel_status(transaction, registered_channels, now)
    return all(s == "closed" for s in status.values())
