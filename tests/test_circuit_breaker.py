"""Tests for B4 — Circuit Breaker (recovery_agent/circuit_breaker.py).

Per notes/PLAN.md's Phase 4 verification step: exactly-15th attempt closes card;
hard decline closes immediately at attempt 1; UPI/netbanking caps trigger at their
configured values; "all closed" signal fires correctly when every registered
channel is closed.
"""

from datetime import datetime, timedelta

from recovery_agent.circuit_breaker import (
    CARD_MAX_ATTEMPTS_PER_WINDOW,
    CARD_MIN_RETRY_SPACING_HOURS,
    NETBANKING_MAX_ATTEMPTS_PER_WINDOW,
    UPI_MAX_ATTEMPTS_PER_WINDOW,
    all_channels_closed,
    get_channel_status,
    is_channel_open,
)
from recovery_agent.models import (
    AttemptOutcome,
    AttemptRecord,
    DeclineCategory,
    MandateChannel,
    TransactionState,
)

NOW = datetime(2026, 1, 1, 12, 0, 0)


def _transaction() -> TransactionState:
    return TransactionState(
        transaction_id="txn-1",
        billing_event_id="evt-1",
        customer_id="cust-1",
        merchant_id="merch-1",
    )


def _add_attempt(
    transaction: TransactionState,
    channel: MandateChannel,
    attempted_at: datetime,
    category: DeclineCategory = DeclineCategory.SOFT,
    outcome: AttemptOutcome = AttemptOutcome.DECLINED,
) -> None:
    transaction.record_attempt(
        AttemptRecord(
            transaction_id=transaction.transaction_id,
            attempt_number=transaction.channel_attempt_count(channel) + 1,
            channel=channel,
            attempted_at=attempted_at,
            chosen_arm="same_channel_same_route",
            outcome=outcome,
            decline_code="51" if channel == MandateChannel.CARD else "SOME_CODE",
            decline_category=category if outcome == AttemptOutcome.DECLINED else None,
        )
    )


def test_card_stays_open_below_visa_threshold():
    transaction = _transaction()
    # Space attempts >=2 days apart to dodge the 24h spacing rule, but all
    # within the 30-day window.
    for i in range(1, CARD_MAX_ATTEMPTS_PER_WINDOW):
        _add_attempt(transaction, MandateChannel.CARD, NOW - timedelta(days=2 * i))
    assert is_channel_open(transaction, MandateChannel.CARD, NOW) is True


def test_card_closes_on_exactly_the_15th_attempt():
    transaction = _transaction()
    for i in range(CARD_MAX_ATTEMPTS_PER_WINDOW):
        _add_attempt(transaction, MandateChannel.CARD, NOW - timedelta(days=i))
    assert transaction.channel_attempt_count(MandateChannel.CARD) == CARD_MAX_ATTEMPTS_PER_WINDOW
    assert is_channel_open(transaction, MandateChannel.CARD, NOW) is False


def test_card_attempts_outside_the_rolling_window_dont_count():
    transaction = _transaction()
    # 14 attempts inside the window, plus several stale ones well outside it.
    for i in range(1, CARD_MAX_ATTEMPTS_PER_WINDOW):
        _add_attempt(transaction, MandateChannel.CARD, NOW - timedelta(days=2 * i))
    for i in range(5):
        _add_attempt(transaction, MandateChannel.CARD, NOW - timedelta(days=60 + i))
    assert is_channel_open(transaction, MandateChannel.CARD, NOW) is True


def test_card_spacing_rule_blocks_retry_within_24h_of_a_soft_decline():
    transaction = _transaction()
    _add_attempt(
        transaction,
        MandateChannel.CARD,
        NOW - timedelta(hours=CARD_MIN_RETRY_SPACING_HOURS - 1),
        category=DeclineCategory.SOFT,
    )
    assert is_channel_open(transaction, MandateChannel.CARD, NOW) is False


def test_card_spacing_rule_allows_retry_after_24h():
    transaction = _transaction()
    _add_attempt(
        transaction,
        MandateChannel.CARD,
        NOW - timedelta(hours=CARD_MIN_RETRY_SPACING_HOURS + 1),
        category=DeclineCategory.SOFT,
    )
    assert is_channel_open(transaction, MandateChannel.CARD, NOW) is True


def test_hard_decline_closes_card_immediately_at_attempt_one():
    transaction = _transaction()
    _add_attempt(transaction, MandateChannel.CARD, NOW, category=DeclineCategory.HARD)
    assert is_channel_open(transaction, MandateChannel.CARD, NOW) is False


def test_hard_decline_closes_upi_immediately_at_attempt_one():
    transaction = _transaction()
    _add_attempt(transaction, MandateChannel.UPI, NOW, category=DeclineCategory.HARD)
    assert is_channel_open(transaction, MandateChannel.UPI, NOW) is False


def test_upi_cap_triggers_at_configured_value():
    transaction = _transaction()
    for i in range(UPI_MAX_ATTEMPTS_PER_WINDOW - 1):
        _add_attempt(transaction, MandateChannel.UPI, NOW - timedelta(hours=i))
    assert is_channel_open(transaction, MandateChannel.UPI, NOW) is True

    _add_attempt(
        transaction, MandateChannel.UPI, NOW - timedelta(hours=UPI_MAX_ATTEMPTS_PER_WINDOW - 1)
    )
    assert transaction.channel_attempt_count(MandateChannel.UPI) == UPI_MAX_ATTEMPTS_PER_WINDOW
    assert is_channel_open(transaction, MandateChannel.UPI, NOW) is False


def test_netbanking_cap_triggers_at_configured_value():
    transaction = _transaction()
    for i in range(NETBANKING_MAX_ATTEMPTS_PER_WINDOW):
        _add_attempt(transaction, MandateChannel.NETBANKING, NOW - timedelta(hours=i))
    assert (
        transaction.channel_attempt_count(MandateChannel.NETBANKING)
        == NETBANKING_MAX_ATTEMPTS_PER_WINDOW
    )
    assert is_channel_open(transaction, MandateChannel.NETBANKING, NOW) is False


def test_get_channel_status_reports_only_registered_channels():
    transaction = _transaction()
    _add_attempt(transaction, MandateChannel.CARD, NOW, category=DeclineCategory.HARD)
    status = get_channel_status(
        transaction, [MandateChannel.CARD, MandateChannel.UPI], NOW
    )
    assert status == {MandateChannel.CARD: "closed", MandateChannel.UPI: "open"}


def test_all_channels_closed_signal_fires_only_when_every_registered_channel_closed():
    transaction = _transaction()
    _add_attempt(transaction, MandateChannel.CARD, NOW, category=DeclineCategory.HARD)
    assert (
        all_channels_closed(transaction, [MandateChannel.CARD, MandateChannel.UPI], NOW)
        is False
    )

    _add_attempt(transaction, MandateChannel.UPI, NOW, category=DeclineCategory.HARD)
    assert (
        all_channels_closed(transaction, [MandateChannel.CARD, MandateChannel.UPI], NOW)
        is True
    )
