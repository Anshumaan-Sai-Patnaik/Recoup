"""Honesty regression tests for C1 — Audit Trail (recovery_agent/audit_trail.py).

A second deliberate, narrow addition to the pytest scope notes/DESIGN.md §5 sets, taken
for the same reason `tests/test_orchestrator.py` was: these guard two defects found by
reading the rendered sentences in Phase 10 task 5, which no component-level test could
have caught — because in both cases every component behaved correctly and it was the
*sentence about them* that was wrong.

ARCHITECTURE.md C1 operation 5 ("never assert a category or reason the system isn't
actually confident of") is the requirement being defended. It is the one property of this
component that fails silently: a wrong sentence looks exactly like a right one, and
judges read this log more closely than anything else in the project. Phase 12 will edit
presentation code with these renderers in reach, so the invariants are pinned here rather
than left to a throwaway script that ran once.

These tests check *claims*, not prose. They assert what a sentence must never say, and
which specific fact it must name — never its exact wording, which is free to improve.
"""

from datetime import datetime

from recovery_agent import bandit, pacing
from recovery_agent.audit_trail import AGENT_BASELINE, AGENT_SMART, AuditTrail, render
from recovery_agent.human_fallback import HumanFallbackEvent
from recovery_agent.models import (
    AttemptOutcome,
    DeclineCategory,
    MandateChannel,
    TransactionStatus,
)
from recovery_agent.orchestrator import (
    AttemptDecision,
    OrchestratorConfig,
    SmartAgentRuntime,
    run_transaction,
)
from recovery_agent.simulator import Simulator, SimulatorConfig

NOW = datetime(2026, 1, 1)


def _decision(**overrides: object) -> AttemptDecision:
    """A plausible Smart Agent round, with two channels genuinely available so the
    bandit clause under test is the multi-arm one."""
    fields: dict[str, object] = dict(
        transaction_id="txn_test",
        attempt_number=1,
        decided_at=NOW,
        context_channel=MandateChannel.CARD,
        context_category=DeclineCategory.SOFT,
        channel_status=((MandateChannel.CARD, "open"), (MandateChannel.UPI, "open")),
        available_arms=(MandateChannel.CARD, MandateChannel.UPI),
        chosen_channel=MandateChannel.UPI,
        chosen_arm="switch_channel",
        rolling_success_rate=0.5,
        baseline_success_rate=0.4,
        health_signal=pacing.SIGNAL_HEALTHY,
        aggressiveness=0.5,
        base_wait_hours=12.0,
        jittered_wait_hours=12.5,
        actual_wait_hours=12.5,
        spacing_rule_applied=False,
        attempted_at=NOW,
        outcome=AttemptOutcome.DECLINED,
        decline_code="51",
        decline_category=DeclineCategory.SOFT,
    )
    fields.update(overrides)
    return AttemptDecision(**fields)  # type: ignore[arg-type]


def test_an_exploring_bandit_is_not_described_as_going_on_past_evidence() -> None:
    """Epsilon-greedy explores a fixed fraction of the time, choosing uniformly at
    random. The sentence used to say the bandit went "on what has worked before" for
    every multi-arm choice, crediting roughly one choice in ten to evidence it did not
    use. The two halves must read as the different reasons they are.
    """
    explored = render(_decision(selection_mode=bandit.SELECTION_EXPLORE))
    exploited = render(_decision(selection_mode=bandit.SELECTION_EXPLOIT))

    assert "random" in explored and "exploration" in explored
    assert "success rate on record" not in explored, (
        "an exploration step chose at random and must not be described as having used "
        f"the bandit's record: {explored}"
    )
    assert "success rate on record" in exploited
    assert "random" not in exploited
    assert explored != exploited


def test_an_unrecognized_code_claims_no_category() -> None:
    """B1's refusal to guess (its operation 3) has to survive all the way into the
    English, not be smoothed over on the last hop."""
    sentence = render(
        _decision(
            decline_code="ZZ9",
            decline_category=None,
            unrecognized_decline_code=True,
        )
    )

    assert "ZZ9" in sentence
    assert "not in the decline-code registry" in sentence
    # The one permitted mention of the words is the disclaimer itself ("no soft/hard
    # category is being claimed"), so strip that before checking nothing else asserts one.
    outcome_clause = sentence.split("The attempt was declined")[-1].replace(
        "soft/hard", ""
    )
    assert "soft" not in outcome_clause and "hard" not in outcome_clause


def test_a_baseline_escalation_never_claims_a_channel_was_closed() -> None:
    """The Baseline Agent has no circuit breaker, so its escalations always carry an
    empty `closed_channels`. Rendering that as "every channel was closed" would hand the
    naive agent a safety story it has no claim to."""
    event = HumanFallbackEvent(
        transaction_id="baseline_txn_test",
        customer_id="customer_1",
        merchant_id="merchant_1",
        occurred_at=NOW,
        attempt_count=10,
        message="please update your payment method",
        closed_channels=(),
    )

    sentence = render(event, AGENT_BASELINE)

    assert "circuit breaker" not in sentence
    assert "no channel was ever checked" in sentence
    assert "cap" in sentence


def test_an_undiagnosable_first_failure_does_not_credit_the_circuit_breaker() -> None:
    """The regression this file was opened for.

    When the *original* billing failure comes back with a code B1 doesn't recognise, the
    Orchestrator escalates immediately — correctly, since it will not make money
    decisions on a diagnosis it doesn't have. But it used to hand B5 every registered
    channel as `closed_channels`, so the audit trail said the circuit breaker had
    permanently closed them all. Nothing was closed. The breaker never ran.

    The honest account is on the transaction's closing line, which carries the real
    `unrecognized_decline_code` reason, and the escalation sentence must not compete with
    it by inventing a different one.
    """
    simulator = Simulator(
        SimulatorConfig(
            seed=1,
            num_customers=1,
            mandate_profile_mix={(MandateChannel.CARD, MandateChannel.UPI): 1.0},
            base_failure_rate=1.0,
        )
    )
    customer = simulator.customers[0]
    merchant = next(
        m for m in simulator.merchants if m.merchant_id == customer.merchant_id
    )
    billing_event = simulator.billing_events[0]
    assert billing_event.billing_event_id in simulator.failed_billing_event_ids

    # A code in no registry — the condition this branch exists for, and one a data-file
    # edit can produce at any time (DESIGN.md keeps the registries as JSON precisely so
    # they can be edited without touching Python).
    simulator.first_attempt_code[billing_event.billing_event_id] = "NOT_A_REAL_CODE"

    config = OrchestratorConfig(seed=0)
    result = run_transaction(
        simulator,
        billing_event,
        customer,
        merchant,
        SmartAgentRuntime.from_config(config),
        config,
    )

    assert result.terminal_reason == "unrecognized_decline_code"
    assert result.transaction.status == TransactionStatus.ESCALATED_TO_HUMAN
    assert result.human_fallback_event is not None
    assert result.human_fallback_event.closed_channels == (), (
        "the circuit breaker closed nothing here; recording channels as closed makes "
        "the audit trail credit a safety mechanism with a decision it never made"
    )

    trail = AuditTrail()
    trail.collect_transaction(result, AGENT_SMART)
    escalation, closing = trail.sentences()

    assert "circuit breaker" not in escalation
    assert "permanently closed" not in escalation
    assert "not in the registry" in closing
