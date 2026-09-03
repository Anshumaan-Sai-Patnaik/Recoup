"""Tests for the Metrics & Evaluation Engine (C3).

**Why this file exists**, given that DESIGN.md §5 scopes `pytest` to the Classifier,
Bandit and Circuit Breaker and leaves the rest to throwaway verification scripts:

The properties guarded here are not "does the arithmetic work" — a throwaway run over a
real batch checks that far better than a fixture can. They are the three places where
this component could produce a number that is *plausible and wrong*, in a way no
component test would ever see because every component would be behaving correctly:

1. **A closure count of 0 for the Baseline Agent.** The Baseline has no Circuit Breaker.
   Reporting `0` would let a reader compare it against the Smart Agent's count as though
   both had counted the same thing, which is the single most misleading cell this table
   could contain. It must be reported as not-applicable.
2. **Counting a temporary spacing wait as a permanent closure.** A card inside its 24h
   spacing window reads as `"closed"` in a decision's `channel_status` while remaining a
   live option; counting those would inflate the headline closure figure with channels
   that were never actually retired.
3. **Scoring the Classifier as a prediction of recoverability.** Its soft/hard call is a
   claim about a *channel* being dead, not about a transaction being lost. A customer
   with a dead card is often still recoverable through UPI, so the naive scoring would
   mark the classifier wrong on exactly the transactions where it was right — quietly,
   in a number that looks respectable either way.
4. **A result quoted with the wrong provenance.** Every number here is true of exactly
   one world. If two runs differing only in failure rate shared a key, two different
   experiments would merge without anyone noticing; if a key changed between processes,
   it would key nothing at all.

Same precedent, and the same reasoning, as `tests/test_audit_trail.py`: a defect in what
a *number* claims is invisible to tests of the mechanisms it describes.
"""

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from recovery_agent import metrics
from recovery_agent.audit_trail import AGENT_BASELINE, AGENT_SMART
from recovery_agent.baseline_agent import (
    BaselineAttemptDecision,
    BaselineConfig,
    BaselineTransactionResult,
)
from recovery_agent.human_fallback import HumanFallbackEvent
from recovery_agent.models import (
    AttemptOutcome,
    AttemptRecord,
    Customer,
    DeclineCategory,
    HiddenTruthRecord,
    MandateChannel,
    MandateRegistration,
    TransactionState,
    TransactionStatus,
)
from recovery_agent.orchestrator import (
    AttemptDecision,
    OrchestratorConfig,
    TransactionResult,
)
from recovery_agent.simulator import SimulatorConfig

START = datetime(2026, 1, 1, 9, 0)
CARD = MandateChannel.CARD
UPI = MandateChannel.UPI


def _transaction(txn_id: str, status: TransactionStatus, attempts: int) -> TransactionState:
    transaction = TransactionState(
        transaction_id=txn_id,
        billing_event_id=txn_id.replace("txn_", "").replace("baseline_", ""),
        customer_id="customer_1",
        merchant_id="merchant_1",
        status=status,
    )
    for number in range(1, attempts + 1):
        transaction.record_attempt(
            AttemptRecord(
                transaction_id=txn_id,
                attempt_number=number,
                channel=CARD,
                attempted_at=START + timedelta(days=number),
                chosen_arm="same_channel_same_route",
                outcome=AttemptOutcome.DECLINED,
                decline_code="51",
            )
        )
    return transaction


def _smart_decision(
    txn_id: str,
    number: int,
    channel_status: tuple[tuple[MandateChannel, str], ...],
    available_arms: tuple[MandateChannel, ...],
) -> AttemptDecision:
    """One Smart Agent round, carrying only the two fields the closure metric reads."""
    moment = START + timedelta(days=number)
    return AttemptDecision(
        transaction_id=txn_id,
        attempt_number=number,
        decided_at=moment,
        context_channel=CARD,
        context_category=DeclineCategory.SOFT,
        channel_status=channel_status,
        available_arms=available_arms,
        chosen_channel=available_arms[0],
        chosen_arm="same_channel_same_route",
        rolling_success_rate=0.0,
        baseline_success_rate=0.0,
        health_signal="warming_up",
        aggressiveness=1.0,
        base_wait_hours=6.0,
        jittered_wait_hours=6.0,
        actual_wait_hours=6.0,
        spacing_rule_applied=False,
        attempted_at=moment,
        outcome=AttemptOutcome.DECLINED,
        decline_code="51",
        decline_category=DeclineCategory.SOFT,
    )


def _baseline_result(txn_id: str, status: TransactionStatus, attempts: int):
    transaction = _transaction(txn_id, status, attempts)
    decisions = [
        BaselineAttemptDecision(
            transaction_id=txn_id,
            attempt_number=number,
            decided_at=START + timedelta(days=number),
            chosen_channel=CARD,
            chosen_arm="same_channel_same_route",
            wait_hours=6.0,
            attempt_cap=10,
            attempted_at=START + timedelta(days=number),
            outcome=AttemptOutcome.DECLINED,
            decline_code="51",
        )
        for number in range(1, attempts + 1)
    ]
    event = None
    if status == TransactionStatus.ESCALATED_TO_HUMAN:
        event = HumanFallbackEvent(
            transaction_id=txn_id,
            customer_id="customer_1",
            merchant_id="merchant_1",
            occurred_at=START + timedelta(days=attempts),
            attempt_count=attempts,
            message="please update your payment method",
        )
    return BaselineTransactionResult(
        transaction=transaction,
        decisions=decisions,
        human_fallback_event=event,
        terminal_reason=(
            "recovered" if status == TransactionStatus.RECOVERED else "attempt_cap_reached"
        ),
    )


def test_baseline_closure_metrics_are_not_applicable_never_zero():
    """The honesty property: an agent with no Circuit Breaker reports no closure count.

    `0` would say it checked and closed nothing. Nothing was ever checked.
    """
    results = [
        _baseline_result("baseline_e1", TransactionStatus.RECOVERED, 3),
        _baseline_result("baseline_e2", TransactionStatus.ESCALATED_TO_HUMAN, 10),
    ]
    summary = metrics.summarise(results)

    assert summary.agent == AGENT_BASELINE
    assert summary.circuit_breaker_consulted is False
    assert summary.channel_closures is None
    assert summary.transactions_with_a_closure is None
    assert summary.closures_by_channel is None
    assert metrics.permanently_closed_channels(results[0]) is None

    # And the rendered comparison cell says so in words, rather than leaving a blank a
    # reader would fill in themselves.
    other = metrics.summarise([_smart_result_recovered()])
    table = metrics.HeadToHead(smart=other, baseline=summary).to_frame()
    cell = table.loc[table["metric"] == "channel_closures", AGENT_BASELINE].iloc[0]
    assert cell == metrics.NOT_APPLICABLE_NO_BREAKER
    assert "0" not in str(cell)


def _smart_result_recovered() -> TransactionResult:
    """A Smart Agent transaction that recovered with every channel still open."""
    txn_id = "txn_e3"
    transaction = _transaction(txn_id, TransactionStatus.RECOVERED, 2)
    decisions = [
        _smart_decision(txn_id, 1, ((CARD, "open"), (UPI, "open")), (CARD, UPI)),
        _smart_decision(txn_id, 2, ((CARD, "open"), (UPI, "open")), (CARD, UPI)),
    ]
    return TransactionResult(
        transaction=transaction, decisions=decisions, terminal_reason="recovered"
    )


def test_a_channel_waiting_out_its_spacing_window_is_not_a_closure():
    """A card inside its 24h spacing window is `"closed"` right now and still a live
    option — it stays in `available_arms`. Counting it would inflate the headline."""
    txn_id = "txn_spacing"
    transaction = _transaction(txn_id, TransactionStatus.RECOVERED, 2)
    decisions = [
        # Card reads "closed" (the spacing timer), but is still an available arm.
        _smart_decision(txn_id, 1, ((CARD, "closed"), (UPI, "open")), (CARD, UPI)),
        _smart_decision(txn_id, 2, ((CARD, "closed"), (UPI, "open")), (CARD, UPI)),
    ]
    result = TransactionResult(
        transaction=transaction, decisions=decisions, terminal_reason="recovered"
    )

    assert metrics.permanently_closed_channels(result) == ()
    assert metrics.summarise([result]).channel_closures == 0


def test_a_channel_dropped_from_the_arms_is_counted_once():
    """A channel the breaker really did retire is counted — once, however many rounds
    it goes on being absent for."""
    txn_id = "txn_closed"
    transaction = _transaction(txn_id, TransactionStatus.ESCALATED_TO_HUMAN, 3)
    decisions = [
        _smart_decision(txn_id, 1, ((CARD, "open"), (UPI, "open")), (CARD, UPI)),
        # Card is gone from the arms from here on: permanently closed.
        _smart_decision(txn_id, 2, ((CARD, "closed"), (UPI, "open")), (UPI,)),
        _smart_decision(txn_id, 3, ((CARD, "closed"), (UPI, "open")), (UPI,)),
    ]
    result = TransactionResult(
        transaction=transaction,
        decisions=decisions,
        human_fallback_event=HumanFallbackEvent(
            transaction_id=txn_id,
            customer_id="customer_1",
            merchant_id="merchant_1",
            occurred_at=START + timedelta(days=4),
            attempt_count=3,
            message="please update your payment method",
            # The final round closes UPI too, and emits no decision of its own — so
            # this event is the only record of that closure.
            closed_channels=(CARD, UPI),
        ),
        terminal_reason="all_channels_closed",
    )

    assert metrics.permanently_closed_channels(result) == (CARD, UPI)
    summary = metrics.summarise([result])
    assert summary.channel_closures == 2
    assert summary.transactions_with_a_closure == 1
    assert summary.closures_by_channel == {"card": 1, "upi": 1}


def test_no_recoveries_reports_none_not_zero_attempts_to_recovery():
    """"Recoveries took 0 attempts" is the wrong reading of "nothing recovered"."""
    summary = metrics.summarise(
        [_baseline_result("baseline_e4", TransactionStatus.ESCALATED_TO_HUMAN, 10)]
    )
    assert summary.recovered == 0
    assert summary.recovery_rate == 0.0
    assert summary.mean_attempts_to_recovery is None
    assert summary.median_attempts_to_recovery is None


def test_mixed_agent_batch_is_rejected():
    """A head-to-head whose two halves can be silently merged is not a comparison."""
    with pytest.raises(ValueError, match="mixed"):
        metrics.summarise(
            [
                _smart_result_recovered(),
                _baseline_result("baseline_e5", TransactionStatus.RECOVERED, 1),
            ]
        )


def test_summary_counts_agree_with_the_transaction_frame():
    """The frame the Dashboard renders and the summary it captions must not disagree."""
    results = [
        _baseline_result("baseline_e6", TransactionStatus.RECOVERED, 2),
        _baseline_result("baseline_e7", TransactionStatus.RECOVERED, 4),
        _baseline_result("baseline_e8", TransactionStatus.ESCALATED_TO_HUMAN, 10),
    ]
    frame = metrics.transaction_frame(results)
    summary = metrics.summarise(results, frame=frame)

    assert summary.agent == AGENT_BASELINE
    assert len(frame) == summary.transactions == 3
    assert int(frame["recovered"].sum()) == summary.recovered == 2
    assert int(frame["attempt_count"].sum()) == summary.total_attempts == 16
    assert summary.mean_attempts_to_recovery == 3.0
    assert summary.terminal_reasons == {"recovered": 2, "attempt_cap_reached": 1}
    assert AGENT_SMART not in frame["agent"].tolist()


# ---------------------------------------------------------------------------
# Against the hidden truth (C3 operations 5-7)
#
# The property guarded here is the third way this component could be plausibly wrong,
# and it is the subtlest of the three: scoring the Classifier as though its soft/hard
# call were a prediction of whether the transaction would recover. It is not. A hard
# decline is a claim about the *channel* being dead, and a customer with a dead card can
# still be recovered through UPI — so the naive scoring would mark the classifier wrong
# on precisely the transactions where it was right, and would do so quietly, in a number
# that looks respectable either way.
# ---------------------------------------------------------------------------


class _FakeSimulator:
    """The smallest thing `against_truth` will accept: a public `customers` list, the
    `hidden_truths` answer key and `first_attempt_category`.

    A stand-in rather than a real `Simulator` because these tests are about *scoring*, and
    generating a real world to get two specific hidden truths would make the setup longer
    than the assertion and would couple the test to the simulator's own rolls.
    """

    def __init__(self, customers, hidden_truths, first_attempt_category):
        self.customers = customers
        self.hidden_truths = hidden_truths
        self.first_attempt_category = first_attempt_category


def _customer(customer_id: str, channels) -> Customer:
    return Customer(
        customer_id=customer_id,
        merchant_id="merchant_1",
        mandates=[
            MandateRegistration(channel=channel, registered_at=START)
            for channel in channels
        ],
    )


def _smart_result(
    txn_id: str,
    customer_id: str,
    status: TransactionStatus,
    attempts: int,
    origin_category: DeclineCategory,
    attempt_channel: MandateChannel = CARD,
) -> TransactionResult:
    transaction = _transaction(txn_id, status, attempts)
    transaction.customer_id = customer_id
    for attempt in transaction.attempts:
        attempt.channel = attempt_channel
    return TransactionResult(
        transaction=transaction,
        decisions=[],
        terminal_reason=(
            "recovered" if status == TransactionStatus.RECOVERED else "all_channels_closed"
        ),
        origin_channel=CARD,
        origin_category=origin_category,
    )


def test_a_hard_call_recovered_by_switching_channel_is_not_a_classifier_error():
    """The subtle one. A hard decline says *this channel* is dead, not *this customer is
    unrecoverable*. Scoring it against the transaction's recoverability would mark the
    classifier wrong exactly when it was right."""
    customer = _customer("customer_1", [CARD, UPI])
    # The answer key agrees the card is dead, but UPI would work.
    truth = HiddenTruthRecord(
        billing_event_id="e_switch",
        is_recoverable=True,
        recoverable_channels=[UPI],
        recoverable_after_seconds=3600.0,
        recoverable_on_attempt_number=1,
    )
    result = _smart_result(
        "txn_e_switch",
        "customer_1",
        TransactionStatus.RECOVERED,
        1,
        DeclineCategory.HARD,
        attempt_channel=UPI,
    )
    result.transaction.billing_event_id = "e_switch"

    simulator = _FakeSimulator(
        [customer], {"e_switch": truth}, {"e_switch": DeclineCategory.HARD}
    )
    scored = metrics.against_truth([result], simulator)
    diagnosis = scored.classifier

    # The transaction was recoverable and was recovered...
    assert scored.recoverable == 1
    assert scored.recall == 1.0
    # ...and the hard call is still scored as correct, because the claim it made — that
    # the card was finished — is what the answer key says too.
    assert diagnosis.hard_calls == 1
    assert diagnosis.hard_calls_on_a_dead_channel == 1
    assert diagnosis.hard_call_channel_precision == 1.0
    # The recoverability figure is reported as the base rate it is, not as an error rate.
    assert diagnosis.recoverable_given_hard == 1.0
    # And the behavioural test: nothing was retried on the channel called dead.
    assert diagnosis.attempts_on_a_dead_origin_channel == 0


def test_retrying_a_channel_the_agent_called_dead_is_counted():
    """The number IDEA.md §4 says this project exists to drive to zero. If the metric
    cannot see the behaviour, the claim that it is zero means nothing."""
    customer = _customer("customer_2", [CARD])
    truth = HiddenTruthRecord(billing_event_id="e_dead", is_recoverable=False)
    result = _smart_result(
        "txn_e_dead",
        "customer_2",
        TransactionStatus.ESCALATED_TO_HUMAN,
        3,
        DeclineCategory.HARD,
        attempt_channel=CARD,
    )
    result.transaction.billing_event_id = "e_dead"

    simulator = _FakeSimulator(
        [customer], {"e_dead": truth}, {"e_dead": DeclineCategory.HARD}
    )
    diagnosis = metrics.against_truth([result], simulator).classifier
    assert diagnosis.attempts_on_a_dead_origin_channel == 3


def test_baseline_has_no_classifier_diagnosis_at_all():
    """`None`, not a diagnosis full of zeros or a 100% score. The Baseline never formed
    an opinion that could be right or wrong."""
    customer = _customer("customer_1", [CARD])
    truth = HiddenTruthRecord(
        billing_event_id="e1",
        is_recoverable=True,
        recoverable_channels=[CARD],
        recoverable_after_seconds=0.0,
        recoverable_on_attempt_number=1,
    )
    result = _baseline_result("baseline_e1", TransactionStatus.RECOVERED, 2)
    result.transaction.billing_event_id = "e1"
    simulator = _FakeSimulator([customer], {"e1": truth}, {"e1": DeclineCategory.SOFT})

    scored = metrics.against_truth([result], simulator)
    assert scored.classifier is None
    # The rest of the against-truth scoring still works for it — only operation 7 is
    # unavailable, and that is the whole point of running the Baseline at all.
    assert scored.recall == 1.0
    assert scored.wasted_attempts == 0


def test_recall_and_waste_split_on_the_answer_key_not_on_the_outcome():
    """Recall counts only genuinely-recoverable transactions; waste counts only
    genuinely-hopeless ones. Mixing the two denominators would let an agent look good by
    failing on transactions that were impossible anyway."""
    customer = _customer("customer_1", [CARD])
    truths = {
        "e_ok": HiddenTruthRecord(
            billing_event_id="e_ok",
            is_recoverable=True,
            recoverable_channels=[CARD],
            recoverable_after_seconds=0.0,
            recoverable_on_attempt_number=1,
        ),
        "e_missed": HiddenTruthRecord(
            billing_event_id="e_missed",
            is_recoverable=True,
            recoverable_channels=[CARD],
            recoverable_after_seconds=0.0,
            recoverable_on_attempt_number=1,
        ),
        "e_hopeless": HiddenTruthRecord(billing_event_id="e_hopeless", is_recoverable=False),
    }
    results = []
    for event_id, status, attempts in (
        ("e_ok", TransactionStatus.RECOVERED, 2),
        ("e_missed", TransactionStatus.ESCALATED_TO_HUMAN, 4),
        ("e_hopeless", TransactionStatus.ESCALATED_TO_HUMAN, 6),
    ):
        result = _smart_result(
            f"txn_{event_id}", "customer_1", status, attempts, DeclineCategory.SOFT
        )
        result.transaction.billing_event_id = event_id
        results.append(result)

    simulator = _FakeSimulator(
        [customer], truths, {event_id: DeclineCategory.SOFT for event_id in truths}
    )
    scored = metrics.against_truth(results, simulator)

    assert (scored.recoverable, scored.unrecoverable) == (2, 1)
    assert scored.recovered_of_recoverable == 1
    assert scored.recall == 0.5
    assert scored.missed_recoverable == 1
    assert scored.missed_by_terminal_reason == {"all_channels_closed": 1}
    # Only the hopeless transaction's attempts count as waste — not the missed one's,
    # which were spent on a recovery that was genuinely available.
    assert scored.wasted_attempts == 6
    assert scored.mean_wasted_attempts == 6.0
    assert scored.wasted_attempt_share == 6 / 12


def test_an_impossible_recovery_is_surfaced_not_assumed_away():
    """`recovered_of_unrecoverable` must be 0 in any sane run, and the metric reports it
    rather than trusting it — a non-zero means the answer key and the fake bank have
    disagreed and every other number is suspect."""
    customer = _customer("customer_1", [CARD])
    truth = HiddenTruthRecord(billing_event_id="e_bad", is_recoverable=False)
    result = _smart_result(
        "txn_e_bad", "customer_1", TransactionStatus.RECOVERED, 1, DeclineCategory.SOFT
    )
    result.transaction.billing_event_id = "e_bad"
    simulator = _FakeSimulator([customer], {"e_bad": truth}, {"e_bad": DeclineCategory.SOFT})

    scored = metrics.against_truth([result], simulator)
    assert scored.recovered_of_unrecoverable == 1


# ---------------------------------------------------------------------------
# The citable results object (C3 operation 8)
#
# The fourth way this component could be plausibly wrong, and the one with the longest
# blast radius: a result that is quoted with the wrong provenance. Every number this
# project produces is true only of one specific world, and the object exists so that the
# world travels with the number instead of in a caption beside it. Two runs that differ
# only in failure rate producing the same key would silently merge two different
# experiments; a key that changed between processes would key nothing at all.
#
# These use a small real run rather than fixtures: the pipeline is milliseconds at this
# size, and the properties being checked are about the packaging of a genuine run.
# ---------------------------------------------------------------------------

SMALL_WORLD = SimulatorConfig(seed=5, num_customers=40, base_failure_rate=0.3)


def test_the_key_identifies_the_whole_configuration_not_just_the_seed():
    """A seed alone does not identify a run. The same seed at a different failure rate is
    a different world, and a citation that cannot tell them apart is not a citation."""
    base_key = metrics.run_key(SMALL_WORLD, OrchestratorConfig(), BaselineConfig())

    assert base_key == metrics.run_key(SMALL_WORLD, OrchestratorConfig(), BaselineConfig())
    assert len(base_key) == 12

    same_seed_other_world = SimulatorConfig(seed=5, num_customers=40, base_failure_rate=0.4)
    assert metrics.run_key(same_seed_other_world, OrchestratorConfig(), BaselineConfig()) != base_key

    # An agent-side change counts too: the same world played differently is a different
    # experiment, even though the customers are identical.
    assert (
        metrics.run_key(SMALL_WORLD, OrchestratorConfig(epsilon=0.5), BaselineConfig())
        != base_key
    )
    assert (
        metrics.run_key(SMALL_WORLD, OrchestratorConfig(), BaselineConfig(max_attempts=3))
        != base_key
    )


def test_the_result_carries_the_configuration_it_actually_ran_with():
    """The provenance is read off the run, never supplied alongside it. A results object
    stamped with a configuration the caller merely asserted would be worse than none."""
    orchestrator = OrchestratorConfig(seed=2, epsilon=0.25)
    baseline = BaselineConfig(max_attempts=4)
    results = metrics.run_evaluation(SMALL_WORLD, orchestrator, baseline)

    assert results.seed == SMALL_WORLD.seed
    assert results.simulator_config["base_failure_rate"] == SMALL_WORLD.base_failure_rate
    assert results.orchestrator_config["epsilon"] == 0.25
    assert results.baseline_config["max_attempts"] == 4
    assert results.key == metrics.run_key(SMALL_WORLD, orchestrator, baseline)

    # Omitted agent configurations are recorded as the defaults that actually ran, not
    # left blank for a reader to guess at.
    defaulted = metrics.run_evaluation(SMALL_WORLD)
    assert defaulted.orchestrator_config == metrics.config_dict(OrchestratorConfig())
    assert defaulted.baseline_config == metrics.config_dict(BaselineConfig())


def test_the_same_configuration_reproduces_the_same_result():
    """The operational meaning of "reproducible": the configuration inside an exported
    result rebuilds that result exactly."""
    first = metrics.run_evaluation(SMALL_WORLD, OrchestratorConfig(seed=2), BaselineConfig())
    second = metrics.run_evaluation(SMALL_WORLD, OrchestratorConfig(seed=2), BaselineConfig())
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_the_honesty_rules_survive_being_exported():
    """The not-applicable cells must not quietly become zeros on the way out of the
    object. An export is where a number stops being read next to its caveats."""
    results = metrics.run_evaluation(SMALL_WORLD, OrchestratorConfig(seed=2), BaselineConfig())
    exported = json.loads(results.to_json())

    assert exported["head_to_head"][AGENT_BASELINE]["channel_closures"] is None
    assert exported["against_truth"][AGENT_BASELINE]["classifier"] is None
    assert exported["against_truth"][AGENT_SMART]["classifier"] is not None

    frame = results.to_frame()
    # The baseline has no classifier section at all, rather than a section of zeros.
    assert frame[(frame["section"] == "classifier") & (frame["agent"] == AGENT_BASELINE)].empty
    closures = frame[
        (frame["section"] == "head_to_head") & (frame["metric"] == "channel_closures")
    ]
    baseline_cell = closures[closures["agent"] == AGENT_BASELINE]["value"].iloc[0]
    assert baseline_cell is None or pd.isna(baseline_cell)


def test_the_export_is_complete_and_stackable():
    """Everything computed is in the table, every row says which run it came from, and
    two runs concatenate without losing that."""
    first = metrics.run_evaluation(SMALL_WORLD, OrchestratorConfig(seed=2), BaselineConfig())
    second = metrics.run_evaluation(
        SimulatorConfig(seed=6, num_customers=40, base_failure_rate=0.3),
        OrchestratorConfig(seed=2),
        BaselineConfig(),
    )

    frame = first.to_frame()
    assert list(frame.columns) == ["run_key", "section", "agent", "metric", "value"]
    assert (frame["run_key"] == first.key).all()
    assert {"head_to_head", "comparison", "against_truth", "classifier", "config"} == set(
        frame["section"]
    )
    # The nested count dicts are flattened rather than dropped.
    assert any(m.startswith("terminal_reasons.") for m in frame["metric"])

    stacked = pd.concat([frame, second.to_frame()], ignore_index=True)
    assert stacked["run_key"].nunique() == 2
    assert len(stacked) == len(frame) + len(second.to_frame())


def test_exports_write_only_when_asked(tmp_path):
    """Both exports return their text and touch the disk only if given a path — Phase 12's
    download button needs the string, not a temp file on the demo machine."""
    results = metrics.run_evaluation(SMALL_WORLD, OrchestratorConfig(seed=2), BaselineConfig())

    assert isinstance(results.to_json(), str)
    assert isinstance(results.to_csv(), str)
    assert not list(tmp_path.iterdir())

    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    json_text = results.to_json(json_path)
    csv_text = results.to_csv(csv_path)

    # newline="" so the line endings pandas chose are read back as written.
    assert json_path.read_text(encoding="utf-8") == json_text
    assert csv_path.open(encoding="utf-8", newline="").read() == csv_text
    assert json.loads(json_path.read_text(encoding="utf-8"))["key"] == results.key
