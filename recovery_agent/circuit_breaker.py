"""B4 — Circuit Breaker.

The safety net: the hard outer boundary nothing above (Bandit, Pacing) is allowed to
cross. Stateless by design — every fact it needs (attempt counts, timestamps, decline
categories) is read straight off a `TransactionState`'s own `AttemptRecord` list rather
than tracked separately, so there is never a second source of truth to go stale
(DESIGN.md B4; see notes/ARCHITECTURE.md Part B4 for the full role description).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple, Optional

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

# The same caps again, keyed per channel and read *without* their rolling window, used
# only by `is_channel_permanently_closed` below — see that function's docstring for why
# a per-transaction cap is treated as spent-for-good rather than as something a
# transaction can wait out.
_LIFETIME_CAPS: dict[MandateChannel, int] = {
    MandateChannel.CARD: CARD_MAX_ATTEMPTS_PER_WINDOW,
    MandateChannel.UPI: UPI_MAX_ATTEMPTS_PER_WINDOW,
    MandateChannel.NETBANKING: NETBANKING_MAX_ATTEMPTS_PER_WINDOW,
}


class PendingDecline(NamedTuple):
    """A decline that has happened but is not (yet) in the transaction's own
    `AttemptRecord` list.

    Exists for one specific case: the **original billing-event failure**. By the
    convention this project settled in Phase 2, that first failed charge is not itself
    an attempt — only the retries that follow it are (see notes/TRACKER.md). But it is
    still a real decline, and B4's rules are supposed to react to it: a card that hard-
    declined on the original charge must be closed before the first retry is ever made,
    and the 24h soft-decline spacing runs from that original decline, not from some
    later attempt.

    Without this, the Circuit Breaker would silently ignore the single most important
    decline in a transaction's life. Every public function below accepts it as an
    optional argument and folds it in exactly as if it were the channel's most recent
    attempt record. Callers with nothing pending pass nothing and get the old behavior.
    """

    channel: MandateChannel
    category: DeclineCategory
    occurred_at: datetime


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


def _last_decline(
    transaction: TransactionState,
    channel: MandateChannel,
    pending_decline: Optional[PendingDecline] = None,
) -> Optional[tuple[DeclineCategory, datetime]]:
    """The most recent decline on `channel` — its category and when it happened —
    considering both this transaction's recorded attempts and any `pending_decline`
    that hasn't been recorded as an attempt yet (see `PendingDecline`).

    Returns None if the channel has never declined on this transaction. Every rule
    below that reasons about "the last thing this channel did" goes through here, so
    the pending decline is folded in at exactly one place rather than in each rule.
    """
    candidates: list[tuple[DeclineCategory, datetime]] = []

    last_attempt = _most_recent_attempt(transaction, channel)
    if last_attempt is not None and last_attempt.decline_category is not None:
        candidates.append((last_attempt.decline_category, last_attempt.attempted_at))
    if pending_decline is not None and pending_decline.channel == channel:
        candidates.append((pending_decline.category, pending_decline.occurred_at))

    if not candidates:
        return None
    return max(candidates, key=lambda c: c[1])


def _card_closed_by_spacing_rule(
    transaction: TransactionState,
    now: datetime,
    pending_decline: Optional[PendingDecline] = None,
) -> bool:
    """Minimum ~24h spacing between soft-decline card retries (ARCHITECTURE.md B4,
    operation 3). If the most recent card decline on this transaction was soft and
    fewer than `CARD_MIN_RETRY_SPACING_HOURS` (simulated) hours have elapsed since
    it, the card channel is temporarily closed — not permanently, just not eligible
    again until the spacing window passes.

    This is the one closure rule that is a *wait* rather than a *wall*, which is why
    `is_channel_permanently_closed` below deliberately excludes it and
    `earliest_next_attempt_at` exists to say when the wait ends.

    A hard decline doesn't trigger this rule on its own; it closes the channel
    outright via operation 5 (`_closed_by_hard_decline`) instead.
    """
    last = _last_decline(transaction, MandateChannel.CARD, pending_decline)
    if last is None or last[0] != DeclineCategory.SOFT:
        return False
    return now - last[1] < timedelta(hours=CARD_MIN_RETRY_SPACING_HOURS)


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


def _closed_by_hard_decline(
    transaction: TransactionState,
    channel: MandateChannel,
    pending_decline: Optional[PendingDecline] = None,
) -> bool:
    """Any channel closes immediately, regardless of attempt count, the moment its
    most recent decline on this transaction came back hard (ARCHITECTURE.md B4,
    operation 5). A hard decline is a permanent signal (e.g. lost/stolen card, mandate
    revoked) — no rolling window or spacing check applies, unlike the soft-decline
    rules above.

    The decline that closes a channel here may be the original billing-event failure
    rather than a recorded retry — hence `pending_decline`. A customer whose only
    registered channel hard-declined on the original charge is escalated to a human
    without a single automated retry, which is correct, and is exactly the
    zero-attempt case `human_fallback._attempt_phrase` is already worded for.
    """
    last = _last_decline(transaction, channel, pending_decline)
    return last is not None and last[0] == DeclineCategory.HARD


def is_channel_open(
    transaction: TransactionState,
    channel: MandateChannel,
    now: datetime,
    pending_decline: Optional[PendingDecline] = None,
) -> bool:
    """Whether `channel` may be attempted **right now**, at `now`, for this
    transaction — the single source of truth combining every closure rule above (Visa
    rule + 24h spacing + assumed UPI/netbanking caps + immediate hard-decline closure).

    This is the authoritative gate: the Orchestrator (D1) checks it immediately before
    submitting any attempt, once its clock has advanced to that attempt's time.
    """
    if _closed_by_hard_decline(transaction, channel, pending_decline):
        return False
    if channel == MandateChannel.CARD:
        if _card_closed_by_visa_rule(transaction, now):
            return False
        if _card_closed_by_spacing_rule(transaction, now, pending_decline):
            return False
        return True
    if _closed_by_assumed_cap(transaction, channel, now):
        return False
    return True


def is_channel_permanently_closed(
    transaction: TransactionState,
    channel: MandateChannel,
    pending_decline: Optional[PendingDecline] = None,
) -> bool:
    """Whether `channel` is finished for this transaction — closed in a way no amount
    of waiting will undo.

    This answers a different question from `is_channel_open`, and the difference is
    what makes the Orchestrator's loop behave sanely. Two of B4's rules are **walls**:
    a hard decline (the channel is dead) and a spent attempt cap. One rule is merely a
    **wait**: the 24h soft-decline spacing on card. Asking `is_channel_open` at the
    moment a decision is made conflates the two, and answering "closed" for a channel
    that is only inside its 24h spacing window would push a card-only customer — 70%
    of the simulated population — straight to human escalation without a single retry,
    when the correct behavior is to wait a day and then retry.

    So: the Bandit (B2) picks among channels that are not permanently closed, the
    Orchestrator never schedules an attempt earlier than `earliest_next_attempt_at`,
    and `is_channel_open` still gates the attempt itself at the moment it fires. No
    attempt can therefore violate a B4 rule, and no transaction gives up early over a
    rule that was only ever a timer.

    On caps being treated as permanent: `_closed_by_assumed_cap` and the Visa rule are
    both *rolling window* rules, so in principle a long-running transaction could age
    attempts out of its window and reopen a channel. For a single transaction's
    recovery journey we take the conservative reading instead — once a channel's cap
    has been spent on this transaction, it stays spent. That is both the safer money
    decision and what guarantees the Orchestrator's loop terminates.
    """
    if _closed_by_hard_decline(transaction, channel, pending_decline):
        return True
    cap = _LIFETIME_CAPS.get(channel)
    return cap is not None and transaction.channel_attempt_count(channel) >= cap


def earliest_next_attempt_at(
    transaction: TransactionState,
    channel: MandateChannel,
    now: datetime,
    pending_decline: Optional[PendingDecline] = None,
) -> datetime:
    """The earliest simulated time at which `channel` may next be attempted.

    `now` if nothing is holding it back; otherwise the moment the card channel's 24h
    soft-decline spacing window expires — the only time-based closure B4 has. The
    Orchestrator uses this as a *floor* under the wait Pacing (B3) computes: the safety
    rule can only ever push an attempt later, never pull it earlier, and Pacing stays
    free to wait longer than the minimum whenever system health says it should.

    Says nothing about whether the channel is permanently closed — check
    `is_channel_permanently_closed` for that. A dead channel has no next attempt time
    at all, and this function would misleadingly hand back one.
    """
    if channel != MandateChannel.CARD:
        return now
    last = _last_decline(transaction, MandateChannel.CARD, pending_decline)
    if last is None or last[0] != DeclineCategory.SOFT:
        return now
    return max(now, last[1] + timedelta(hours=CARD_MIN_RETRY_SPACING_HOURS))


def get_channel_status(
    transaction: TransactionState,
    registered_channels: list[MandateChannel],
    now: datetime,
    pending_decline: Optional[PendingDecline] = None,
) -> dict[MandateChannel, str]:
    """The Bandit's (B2) required per-attempt input: for every channel this customer
    has registered, whether it is currently `"open"` (further attempts permitted) or
    `"closed"` (must not be attempted again right now) — ARCHITECTURE.md B4, operation
    6. Only registered channels are reported; a channel the customer never had on file
    is meaningless to report a status for.
    """
    return {
        channel: "open"
        if is_channel_open(transaction, channel, now, pending_decline)
        else "closed"
        for channel in registered_channels
    }


def all_channels_closed(
    transaction: TransactionState,
    registered_channels: list[MandateChannel],
    now: datetime,
    pending_decline: Optional[PendingDecline] = None,
) -> bool:
    """Every one of this customer's registered channels is closed *at this instant*,
    including any that are only waiting out a spacing window. Purely derived from
    `get_channel_status` (ARCHITECTURE.md B4, operation 7).

    Note for callers deciding whether to give up: this is not the Human Fallback (B5)
    trigger — `all_channels_permanently_closed` is. A transaction can have every
    channel "closed" here and still be perfectly recoverable an hour later.
    """
    status = get_channel_status(transaction, registered_channels, now, pending_decline)
    return all(s == "closed" for s in status.values())


def all_channels_permanently_closed(
    transaction: TransactionState,
    registered_channels: list[MandateChannel],
    pending_decline: Optional[PendingDecline] = None,
) -> bool:
    """Every registered channel is finished for this transaction — the real trigger
    condition for routing to Human Fallback (B5), per ARCHITECTURE.md B4 operation 7.

    Never a count-based trigger of its own: it is entirely derived from the per-channel
    rules, so B5 fires exactly when the automated options have genuinely run out, and
    not one attempt sooner.
    """
    return all(
        is_channel_permanently_closed(transaction, channel, pending_decline)
        for channel in registered_channels
    )
