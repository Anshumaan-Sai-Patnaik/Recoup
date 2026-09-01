"""Tests for B2 — Bandit (recovery_agent/bandit.py).

Per notes/PLAN.md's Phase 5 verification step: never selects a circuit-breaker-closed
arm; stats update correctly after a recorded outcome; arm distribution shifts toward a
manually-seeded "winning" arm over many synthetic trials.
"""

import random
from datetime import datetime, timedelta

import pytest

from recovery_agent.bandit import (
    ArmStats,
    BanditStatsPool,
    available_arms,
    context_key,
    select_arm,
)
from recovery_agent.models import (
    AttemptOutcome,
    AttemptRecord,
    Customer,
    DeclineCategory,
    MandateChannel,
    MandateRegistration,
    TransactionState,
)

NOW = datetime(2026, 1, 1, 12, 0, 0)


def _customer(*channels: MandateChannel) -> Customer:
    return Customer(
        customer_id="cust-1",
        merchant_id="merch-1",
        mandates=[
            MandateRegistration(channel=channel, registered_at=NOW - timedelta(days=100))
            for channel in channels
        ],
    )


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
    category: DeclineCategory = DeclineCategory.HARD,
) -> None:
    transaction.record_attempt(
        AttemptRecord(
            transaction_id=transaction.transaction_id,
            attempt_number=transaction.channel_attempt_count(channel) + 1,
            channel=channel,
            attempted_at=attempted_at,
            chosen_arm="same_channel_same_route",
            outcome=AttemptOutcome.DECLINED,
            decline_code="51" if channel == MandateChannel.CARD else "SOME_CODE",
            decline_category=category,
        )
    )


class TestNeverSelectsAClosedArm:
    def test_select_arm_only_ever_returns_an_open_arm(self) -> None:
        # Card hard-declines once, which closes it immediately (B4) — only UPI
        # should ever remain in available_arms, and select_arm must never return
        # card regardless of how many trials or how it's seeded.
        customer = _customer(MandateChannel.CARD, MandateChannel.UPI)
        transaction = _transaction()
        _add_attempt(transaction, MandateChannel.CARD, NOW, category=DeclineCategory.HARD)

        arms = available_arms(customer, transaction, NOW)
        assert arms == [MandateChannel.UPI]

        pool = BanditStatsPool()
        ctx = context_key(MandateChannel.CARD, DeclineCategory.HARD)
        rng = random.Random(0)
        for _ in range(200):
            chosen = select_arm(arms, ctx, pool, rng, epsilon=0.5)
            assert chosen == MandateChannel.UPI

    def test_select_arm_raises_on_no_available_arms(self) -> None:
        pool = BanditStatsPool()
        ctx = context_key(MandateChannel.CARD, DeclineCategory.HARD)
        with pytest.raises(ValueError):
            select_arm([], ctx, pool, random.Random(0))


class TestStatsUpdateAfterOutcome:
    def test_record_outcome_accumulates_attempts_and_successes(self) -> None:
        pool = BanditStatsPool()
        ctx = context_key(MandateChannel.CARD, DeclineCategory.SOFT)

        assert pool.get_stats(ctx, MandateChannel.CARD) == ArmStats()

        pool.record_outcome(ctx, MandateChannel.CARD, True)
        pool.record_outcome(ctx, MandateChannel.CARD, False)
        pool.record_outcome(ctx, MandateChannel.CARD, True)

        stats = pool.get_stats(ctx, MandateChannel.CARD)
        assert stats.attempts == 3
        assert stats.successes == 2
        assert stats.success_rate == pytest.approx(2 / 3)

    def test_record_outcome_does_not_affect_other_contexts_or_channels(self) -> None:
        pool = BanditStatsPool()
        card_soft = context_key(MandateChannel.CARD, DeclineCategory.SOFT)
        card_hard = context_key(MandateChannel.CARD, DeclineCategory.HARD)

        pool.record_outcome(card_soft, MandateChannel.CARD, True)

        assert pool.get_stats(card_hard, MandateChannel.CARD) == ArmStats()
        assert pool.get_stats(card_soft, MandateChannel.UPI) == ArmStats()

    def test_get_stats_is_read_only(self) -> None:
        pool = BanditStatsPool()
        ctx = context_key(MandateChannel.UPI, DeclineCategory.SOFT)

        pool.get_stats(ctx, MandateChannel.UPI)  # a mere read

        pool.record_outcome(ctx, MandateChannel.CARD, True)
        # The read above must not have created a phantom UPI entry that the real
        # write above would then be missing from.
        assert pool.get_stats(ctx, MandateChannel.UPI) == ArmStats()


class TestDistributionShiftsTowardWinningArm:
    def test_select_arm_favors_the_seeded_winning_arm_over_many_trials(self) -> None:
        pool = BanditStatsPool()
        ctx = context_key(MandateChannel.CARD, DeclineCategory.SOFT)

        # Seed lopsided prior evidence: UPI looks much better than card.
        for _ in range(20):
            pool.record_outcome(ctx, MandateChannel.CARD, False)
        for _ in range(20):
            pool.record_outcome(ctx, MandateChannel.UPI, True)

        rng = random.Random(123)
        arms = [MandateChannel.CARD, MandateChannel.UPI]
        counts = {MandateChannel.CARD: 0, MandateChannel.UPI: 0}
        for _ in range(2000):
            chosen = select_arm(arms, ctx, pool, rng, epsilon=0.1)
            counts[chosen] += 1

        # ~90% exploit (always UPI) + ~10% explore (uniform over 2 arms, so ~5%
        # more UPI) — UPI should heavily dominate; card only shows up via explore.
        assert counts[MandateChannel.UPI] > counts[MandateChannel.CARD]
        assert counts[MandateChannel.UPI] / 2000 > 0.85
        assert counts[MandateChannel.CARD] > 0  # explore still gives card a look-in

    def test_select_arm_is_reproducible_for_a_given_seed(self) -> None:
        pool = BanditStatsPool()
        ctx = context_key(MandateChannel.CARD, DeclineCategory.SOFT)
        pool.record_outcome(ctx, MandateChannel.CARD, True)
        arms = [MandateChannel.CARD, MandateChannel.UPI]

        seq_a = [select_arm(arms, ctx, pool, random.Random(42)) for _ in range(30)]
        seq_b = [select_arm(arms, ctx, pool, random.Random(42)) for _ in range(30)]
        assert seq_a == seq_b
