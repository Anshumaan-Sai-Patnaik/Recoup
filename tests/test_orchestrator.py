"""Regression tests for D1 — Orchestrator (recovery_agent/orchestrator.py).

A deliberate, narrow addition to the pytest scope notes/DESIGN.md §5 sets (which covers
B1, B2 and B4, on the grounds that those are the components an incorrect implementation
would be *dangerous* on). These tests exist because Phase 9's head-to-head surfaced a
sequencing bug in this file that was, by construction, invisible to every component-level
test: the Circuit Breaker was correct in isolation and the Orchestrator simply stopped
telling it the truth after the first round.

The bug: `run_transaction` used one `pending` variable both as the Bandit's context (which
must move to the latest decline each round) and as the only surviving record of the
original billing-event failure for the Circuit Breaker (which must not move, since the
original charge is deliberately never an `AttemptRecord`). Overwriting it meant a card
that hard-declined on the original charge looked open again the moment the agent switched
channels, and got retried — the exact fraud-flag trap notes/IDEA.md §4 is written about.

DESIGN.md's own stated reason for testing B1/B2/B4 ("money-related decisions") applies
here squarely, so the scope deviation is recorded in notes/TRACKER.md rather than
silently taken.
"""

from recovery_agent.models import (
    AttemptOutcome,
    DeclineCategory,
    MandateChannel,
    TransactionStatus,
)
from recovery_agent.orchestrator import (
    OrchestratorConfig,
    SmartAgentRuntime,
    run_transaction,
)
from recovery_agent.simulator import Simulator, SimulatorConfig


def _hard_card_world() -> tuple[Simulator, object, object, object]:
    """A one-customer world engineered so the trap is unavoidable if the bug is present.

    The customer holds card + UPI. Their original charge is on card (the primary,
    first-registered channel) and hard-declines. Nothing is recoverable, so the loop
    keeps running for several rounds on UPI — which is precisely the window in which a
    forgotten card hard-decline would let the Bandit pick card again.

    Built from real `SimulatorConfig` knobs rather than hand-injected state, so the test
    exercises the same construction path a real run does:
      - `mandate_profile_mix` forces card+UPI, card first.
      - `base_failure_rate=1.0` forces the billing event to fail.
      - `soft_failure_ratio=0.0` forces that failure to be HARD.
      - `hard_recoverable_via_switch_ratio=0.0` makes UPI never recover, so the loop
        cannot end early by succeeding.
    """
    simulator = Simulator(
        SimulatorConfig(
            seed=1,
            num_customers=1,
            mandate_profile_mix={(MandateChannel.CARD, MandateChannel.UPI): 1.0},
            base_failure_rate=1.0,
            soft_failure_ratio=0.0,
            hard_recoverable_via_switch_ratio=0.0,
        )
    )
    customer = simulator.customers[0]
    merchant = next(m for m in simulator.merchants if m.merchant_id == customer.merchant_id)
    billing_event = simulator.billing_events[0]

    # Guard the fixture itself: if any of the knobs above stop producing this shape, the
    # tests below would silently start passing for the wrong reason.
    assert billing_event.billing_event_id in simulator.failed_billing_event_ids
    assert customer.primary_channel() == MandateChannel.CARD
    assert simulator.first_attempt_category[billing_event.billing_event_id] == (
        DeclineCategory.HARD
    )
    assert not simulator.hidden_truths[billing_event.billing_event_id].is_recoverable

    return simulator, billing_event, customer, merchant


def _run() -> object:
    simulator, billing_event, customer, merchant = _hard_card_world()
    config = OrchestratorConfig(seed=0)
    return run_transaction(
        simulator,
        billing_event,
        customer,
        merchant,
        SmartAgentRuntime.from_config(config),
        config,
    )


def test_hard_declined_channel_is_never_retried_after_switching_channels() -> None:
    """The regression itself: a channel that hard-declined on the *original* charge must
    stay closed for the whole transaction, not just for the first round."""
    result = _run()

    card_attempts = [
        a for a in result.transaction.attempts if a.channel == MandateChannel.CARD
    ]
    assert card_attempts == [], (
        "card hard-declined on the original billing charge and must never be attempted; "
        f"got attempts at {[a.attempted_at.isoformat() for a in card_attempts]}"
    )

    # The transaction must genuinely have run for several rounds on the other channel —
    # otherwise it could pass simply by never getting far enough to make the mistake.
    assert len(result.transaction.attempts) >= 2
    assert all(a.channel == MandateChannel.UPI for a in result.transaction.attempts)
    assert all(a.outcome == AttemptOutcome.DECLINED for a in result.transaction.attempts)


def test_exhausting_every_channel_escalates_to_a_human() -> None:
    """With card closed by the original hard decline and UPI worked down to its cap,
    every registered channel is closed and B5 ends the journey (ARCHITECTURE.md B4
    operation 7 -> B5)."""
    result = _run()

    assert result.transaction.status == TransactionStatus.ESCALATED_TO_HUMAN
    assert result.terminal_reason == "all_channels_closed"
    assert result.human_fallback_event is not None
    # The "because" recorded on the terminal event must name card, which is only true if
    # the original hard decline was still remembered at the moment of escalation.
    assert MandateChannel.CARD in result.human_fallback_event.closed_channels


def test_bandit_context_still_follows_the_latest_decline() -> None:
    """The other half of the fix: `origin_decline` must be frozen, but `context_decline`
    must still move, or the Bandit would spend the whole transaction learning against
    the original failure's context instead of the one it is actually reacting to."""
    result = _run()

    contexts = [(d.context_channel, d.context_category) for d in result.decisions]
    assert contexts[0] == (MandateChannel.CARD, DeclineCategory.HARD)
    assert all(c == (MandateChannel.UPI, DeclineCategory.SOFT) for c in contexts[1:])
