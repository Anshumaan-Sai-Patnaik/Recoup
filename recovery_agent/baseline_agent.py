"""C2 — Baseline Agent.

The naive strawman the Smart Agent is measured against, and the control group the
entire pitch rests on: run this and the Smart Agent (D1, wiring B1-B5) over the *same*
seeded Simulator population, and whatever difference shows up in the outcomes is
attributable to the agent rather than to luck in the world each was handed
(notes/ARCHITECTURE.md Part C2; notes/IDEA.md §11).

Deliberately dumb, by design — that is a feature here, not a shortcut. Every point
where the Smart Agent consults a component, this file substitutes a fixed constant, or
nothing at all:

    Smart Agent (D1 asks ...)              Baseline Agent (C2 instead ...)
    -------------------------------------- --------------------------------------
    B1 Classifier  "soft or hard?"          nothing. The decline code is recorded
                                            but never read, so the same channel is
                                            retried whether the bank said "try
                                            later" or "this card is dead".
    B2 Bandit      "which channel?"         `select_channel` — always the primary
                                            one, every attempt, forever.
    B3 Pacing      "how long to wait?"      `wait_hours` — one fixed interval, with
                                            no AIMD and no jitter.
    B4 Circuit Br. "is this channel        nothing. A plain attempt cap stands in
                    still permitted?"       for every real safety rule.
    B5 Human Fallb. "escalate"              reused unchanged, once the cap is spent.

Built as a *variant of D1's loop shape* rather than as a separate system, per
DESIGN.md's C2 row — that is what guarantees it cannot accidentally acquire an
advantage the Smart Agent doesn't have (a different clock, a different population, a
different notion of what counts as an attempt).

Complete as of Phase 9 (all three tasks): the retry policy (which channel it picks, and
when), the stopping policy (a flat attempt cap, then the generic human nudge) wired into
`run_transaction`, the `run_batch` that applies it across a whole seeded population, and
`run_paired_batch` — the one call that runs both agents over the same world and is what
makes Phase 11's comparison a controlled experiment rather than two unrelated numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Generator, Iterator, Optional, Union

from recovery_agent.human_fallback import (
    HumanFallbackEvent,
    is_terminal,
    run_human_fallback,
)
from recovery_agent.models import (
    AttemptOutcome,
    AttemptRecord,
    BillingEvent,
    Customer,
    MandateChannel,
    Merchant,
    TransactionState,
    TransactionStatus,
)
from recovery_agent.orchestrator import (
    ARM_SAME_CHANNEL,
    OrchestratorConfig,
    TransactionResult,
)
from recovery_agent.orchestrator import iter_batch as iter_smart_batch
from recovery_agent.simulator import SimulatedClock, Simulator, SimulatorConfig

BASELINE_RETRY_INTERVAL_HOURS = 6.0
"""The flat wait between one attempt and the next — the baseline's entire answer to
"when?", standing in for the whole of B3.

Six hours is "a dunning job that runs four times a day", which is a real and common
naive setup rather than a made-up one. It also breaches the ~24h soft-decline retry
spacing guidance (IDEA.md §9) on *every single retry*, which is the specific naive
instinct IDEA.md §4 is written about: retry the same card, soon, repeatedly.
"""

BASELINE_MAX_ATTEMPTS = 10
"""How many retries the baseline makes before handing over to a human.

Deliberately set *below* the commonly-reported Visa cap of 15 attempts per rolling
30-day window. That is a calibration choice worth stating out loud, because it is the
difference between an honest control group and a caricature: a baseline configured to
blow straight through a published network limit would make the Smart Agent look good by
comparison to something nobody actually ships. At 10, the baseline stays inside the
hard cap, and what it still gets wrong is real and specific — it breaches the 24h
spacing guidance every retry, and it spends all ten attempts on a channel even when the
very first decline was a *hard* one, because with no classifier it cannot tell the
difference.
"""

BASELINE_ARM = ARM_SAME_CHANNEL
"""The arm label written onto every `AttemptRecord` this agent produces.

Imported from the Orchestrator rather than redefined, so that both agents' attempt
records speak the same vocabulary and the Metrics engine (C3, Phase 11) can read them
side by side without a translation step. It is also, by itself, an accurate summary of
this agent: there is only ever one arm.
"""


@dataclass
class BaselineConfig:
    """The baseline's fixed constants, gathered in one place.

    Every field here is a *policy* number, unlike `OrchestratorConfig`, whose fields are
    sequencing knobs for a loop whose real decisions live inside B1-B5. That contrast is
    the point: the Smart Agent's behaviour is not configurable in this way because it is
    computed per attempt from live state, while the baseline's behaviour is fully
    described by these four numbers.

    """

    retry_interval_hours: float = BASELINE_RETRY_INTERVAL_HOURS
    """Flat wait between attempts — see `BASELINE_RETRY_INTERVAL_HOURS`."""

    max_attempts: int = BASELINE_MAX_ATTEMPTS
    """Attempt cap before the human nudge — see `BASELINE_MAX_ATTEMPTS`."""

    recovery_horizon_days: float = 30.0
    """The same billing-cycle-length horizon the Orchestrator uses, carried here purely
    so that `ABANDONED` means the identical thing on both sides of the Phase 11
    comparison ("ran out of time", as against `ESCALATED_TO_HUMAN`'s "ran out of
    options").

    At the default settings it never binds — ten attempts six hours apart finish inside
    three days — so it is a guard rail rather than a behaviour. It only starts to matter
    if someone reconfigures the baseline to something slow enough to outlive the billing
    cycle it is trying to recover, and at that point silently running past the next
    charge would be the bug, not the horizon.
    """

    transaction_id_prefix: str = "baseline"
    """Prefixes this agent's transaction ids, mirroring
    `OrchestratorConfig.transaction_id_prefix` (which defaults to `"txn"`), so that when
    the Metrics engine holds both runs at once every transaction says plainly which
    agent produced it."""


def select_channel(customer: Customer) -> MandateChannel:
    """The baseline's answer to "which channel?" — always the customer's primary
    (first-registered) one (ARCHITECTURE.md C2, operation 1).

    This is the deliberate replacement for the Bandit (B2). Compare the signatures and
    the difference *is* the experiment: `bandit.select_arm` takes the set of channels the
    Circuit Breaker still permits, a context key describing the failure, a pool of
    evidence gathered across every customer and merchant in the batch, and an explore
    rate. This takes a customer, and reads one field off them.

    The channel it returns is the same one the original billing charge already failed
    on, which is exactly the behaviour under test: IDEA.md §4's naive instinct is not
    "pick badly among options", it is "never consider that there were options". A
    customer with a UPI mandate sitting unused on file gets their dead card retried ten
    times regardless.

    Route is not modelled anywhere in this build (the Simulator's Hidden Truth varies
    outcomes by channel, not by route within a channel — see notes/TRACKER.md's Phase 2
    deviation), so "same channel, same route" reduces to "same channel" here for the
    baseline exactly as it does for the Smart Agent.
    """
    return customer.primary_channel()


def wait_hours(config: BaselineConfig) -> float:
    """The baseline's answer to "how long until the next attempt?" — the configured
    constant, every time (ARCHITECTURE.md C2, operation 2).

    This is the deliberate replacement for Pacing (B3), and the signature is again the
    evidence. AIMD needs the system's rolling outcome window to decide whether to ease
    off or speed up, and jitter needs a seeded `random.Random` to spread simultaneous
    retries apart; this needs neither, because it adapts to nothing and coordinates with
    nothing. Two hundred subscriptions failing in the same bank outage will, under this
    policy, retry in one synchronised wave six hours later — the thundering herd
    IDEA.md §8c describes, reproduced faithfully rather than avoided.

    It takes `config` rather than reading `BASELINE_RETRY_INTERVAL_HOURS` directly so
    that a run can be configured without reaching into module state, and so this stays
    the single seam to change if the fixed interval is ever swapped for
    ARCHITECTURE.md C2 operation 2's other sanctioned option, a flat exponential backoff.
    """
    return config.retry_interval_hours


def next_attempt_at(now: datetime, config: BaselineConfig) -> datetime:
    """When the next attempt fires, given the current simulated time.

    The clock this reads from is the Orchestrator's per-transaction `SimulatedClock`
    convention (a clock started at *this* billing event's scheduled time, not one global
    clock shared across a sequentially-processed batch — see notes/TRACKER.md's Phase 8
    deviation for why that distinction silently decides whether the Simulator thinks
    every transaction is instantly recoverable). `run_transaction` below follows that
    same convention: a control group that measured time differently from the thing it is
    controlling for would not be a control group.

    Note what has no equivalent here: the Smart Agent puts the Circuit Breaker's 24h
    card spacing rule underneath Pacing's number as a *floor*, so a safety rule can push
    an attempt later but never pull it earlier. The baseline consults no such floor, so
    this time is final — which is precisely how it ends up retrying a card four times a
    day.
    """
    return now + timedelta(hours=wait_hours(config))


BASELINE_ATTEMPT_DECISION_EVENT_TYPE = "baseline_attempt_decision"
"""The `event_type` tag on every per-attempt record this agent emits, distinct from the
Orchestrator's `"attempt_decision"`.

The Audit Trail (C1) picks a renderer per event type, and these two need different
sentences: the Smart Agent's reads "retrying via UPI in 6h because ...", while this
agent's honest sentence is closer to "retrying via card in 6h because that is what we
always do." Tagging at the source is how C1 tells them apart without guessing from an
event's shape — the same convention `human_fallback.HUMAN_FALLBACK_EVENT_TYPE` set.
"""

TERMINAL_RECOVERED = "recovered"
TERMINAL_ATTEMPT_CAP_REACHED = "attempt_cap_reached"
TERMINAL_HORIZON_EXCEEDED = "recovery_horizon_exceeded"
"""Why a baseline transaction's loop stopped, as short machine-readable tags (C1 renders
the English, so this file never asserts prose it would have to keep in sync).

Worth reading against the Orchestrator's equivalents, because Phase 11 counts these and
the same *status* is reached here by a different *route*:

- `TERMINAL_ATTEMPT_CAP_REACHED` ends in `ESCALATED_TO_HUMAN`, and for this agent it is
  the ordinary ending rather than an exceptional one. The Smart Agent only escalates
  when the Circuit Breaker has genuinely closed every registered channel; this agent
  escalates because a counter ran out. Both are honestly "we gave up and asked a human",
  which is what makes them comparable (ARCHITECTURE.md C3, operation 3) — but the reason
  behind the number is not the same, and the audit trail should say so.
- `TERMINAL_HORIZON_EXCEEDED` ends in `ABANDONED`, matching the Orchestrator's meaning
  exactly ("ran out of time", as against escalation's "ran out of options"). At the
  default settings it never fires; see `BaselineConfig.recovery_horizon_days`.
- The Orchestrator's `"all_channels_closed"` and `"unrecognized_decline_code"` have no
  equivalent here, because this agent consults neither the Circuit Breaker nor the
  Classifier. Their absence from a baseline run is a finding, not a gap.
"""


@dataclass(frozen=True)
class BaselineAttemptDecision:
    """One round of the baseline's loop, recorded as facts.

    The deliberately thin counterpart to `orchestrator.AttemptDecision`. Put the two
    side by side and the difference is the entire experiment: the Smart Agent's record
    carries the channel status the Circuit Breaker reported, the arms that were
    available, the context key the Bandit keyed on, the system's rolling success rate,
    its baseline rate, the health signal, the aggressiveness dial, the base wait, the
    jittered wait and whether a safety rule pushed the attempt later. This one carries
    the channel (always the same), the wait (always the same), and the count.

    That is not laziness in the record — it is the record being honest. There is no
    aggressiveness value to log because no such number was ever computed, and inventing
    a plausible-looking one so the two exports have matching columns would be exactly
    the dishonesty the Audit Trail exists to prevent (ARCHITECTURE.md C1, operation 5).

    Frozen, like every other event in this project: a decision record that can be edited
    after the fact isn't a record.
    """

    transaction_id: str
    attempt_number: int
    decided_at: datetime
    chosen_channel: MandateChannel
    chosen_arm: str
    wait_hours: float
    attempt_cap: int
    attempted_at: datetime
    outcome: AttemptOutcome
    decline_code: Optional[str] = None
    event_type: str = BASELINE_ATTEMPT_DECISION_EVENT_TYPE

    def to_dict(self) -> dict[str, Any]:
        """A flat, JSON/CSV-friendly row for C1's `pandas` export.

        Column names deliberately match `orchestrator.AttemptDecision.to_dict()` wherever
        the same fact exists in both, so the two agents' runs concatenate into one frame
        with no translation step. Where a fact exists only for the Smart Agent, this row
        simply omits it and pandas leaves the cell empty — which reads correctly, because
        the baseline genuinely has no value for it.

        `decline_category` is the one exception: it is emitted explicitly as `None`
        rather than omitted, so the column is visibly blank on every baseline row instead
        of quietly absent. That blank is the point — this agent has no classifier, so it
        never formed an opinion about whether a decline was temporary or permanent, and
        the export should show that rather than hide it.
        """
        return {
            "event_type": self.event_type,
            "transaction_id": self.transaction_id,
            "attempt_number": self.attempt_number,
            "decided_at": self.decided_at.isoformat(),
            "chosen_channel": self.chosen_channel.value,
            "chosen_arm": self.chosen_arm,
            "actual_wait_hours": self.wait_hours,
            "attempt_cap": self.attempt_cap,
            "attempted_at": self.attempted_at.isoformat(),
            "outcome": self.outcome.value,
            "decline_code": self.decline_code,
            "decline_category": None,
        }


@dataclass
class BaselineTransactionResult:
    """Everything one baseline transaction's journey produced.

    Structurally the same four fields as `orchestrator.TransactionResult` — final state,
    per-attempt decisions, the terminal human-fallback event, and a reason tag — so the
    Metrics engine (C3, Phase 11) can read both agents' output through one code path.
    It is a separate class only because `decisions` holds a different (thinner) record
    type, and annotating it as the Orchestrator's would be a lie about its contents.
    """

    transaction: TransactionState
    decisions: list[BaselineAttemptDecision] = field(default_factory=list)
    human_fallback_event: Optional[HumanFallbackEvent] = None
    terminal_reason: str = ""


def run_transaction(
    simulator: Simulator,
    billing_event: BillingEvent,
    customer: Customer,
    merchant: Merchant,
    config: Optional[BaselineConfig] = None,
) -> BaselineTransactionResult:
    """Run the naive recovery journey for one failed billing event
    (ARCHITECTURE.md C2, operations 1-4).

    The loop, per round: have we spent the cap? -> if so, hand to Human Fallback and stop
    -> otherwise wait the fixed interval -> submit the attempt on the same channel as
    always -> record it -> approved ends the journey, declined goes round again.

    Read that against the Orchestrator's twelve-operation round and what is *missing* is
    the whole point: nothing is diagnosed, nothing is asked for permission, nothing is
    chosen, nothing adapts, and nothing is learned from. The outcome is written down and
    then thrown away.

    Three things are held identical to the Smart Agent on purpose, because a control
    group that differed from the thing it controls for in any of them would invalidate
    the comparison rather than inform it:

    - **The clock.** A per-transaction `SimulatedClock` started at this billing event's
      own `scheduled_at`, exactly as `orchestrator.run_transaction` does. A batch is
      processed one transaction at a time, so a single shared clock would leave later
      transactions starting weeks after their own billing event — and since the Simulator
      measures recoverability as elapsed time since `scheduled_at`, they would all look
      instantly recoverable (see notes/TRACKER.md's Phase 8 deviation).
    - **What counts as an attempt.** The original billing-event failure is not itself an
      `AttemptRecord`; only the retries after it are (the Phase 2 convention). This is
      what `HiddenTruthRecord.recoverable_on_attempt_number` is measured against, so
      counting differently here would quietly hand one agent easier targets.
    - **Ignoring the Simulator's own category.** `respond_to_attempt` returns the true
      soft/hard category alongside the decline code, and this agent discards it. That
      discipline matters even more here than in the Orchestrator: the Smart Agent has to
      earn its diagnosis through the Classifier, so letting the fake bank hand this agent
      the answer for free would give the *baseline* an advantage the real system does not
      have, and turn the comparison upside down.

    Assumes `billing_event` actually failed — `run_batch` below is responsible for
    filtering, exactly as the Orchestrator's does.
    """
    config = config or BaselineConfig()

    transaction = TransactionState(
        transaction_id=f"{config.transaction_id_prefix}_{billing_event.billing_event_id}",
        billing_event_id=billing_event.billing_event_id,
        customer_id=customer.customer_id,
        merchant_id=customer.merchant_id,
    )
    result = BaselineTransactionResult(transaction=transaction)

    clock = SimulatedClock(billing_event.scheduled_at)
    horizon = billing_event.scheduled_at + timedelta(days=config.recovery_horizon_days)

    # Chosen once, before the loop, and never revisited — which is the single line that
    # most distinguishes this agent from the Smart one. The Bandit is re-consulted before
    # *every* attempt (IDEA.md 8a is emphatic that a transaction's journey can go
    # card -> UPI -> card); this agent decides once and then stops thinking.
    #
    # Note also what is never read: `simulator.first_attempt_code[...]`, the decline code
    # the original charge came back with. The Orchestrator's first act is to hand that to
    # the Classifier. Here it is simply not looked at, so a customer whose card came back
    # "lost or stolen" gets the same treatment as one who was briefly short of funds.
    channel = select_channel(customer)

    while not is_terminal(transaction):
        now = clock.now()

        # ARCHITECTURE.md C2, operations 3-4 — the entire "know your limits" story for
        # this agent: a bare count, standing in for the Visa rolling-window rule, the
        # 24h spacing rule, the assumed UPI/netbanking caps, and immediate closure on a
        # hard decline. It stops at the right *sort* of moment for entirely the wrong
        # reason, which is precisely what Phase 11 is set up to measure.
        if len(transaction.attempts) >= config.max_attempts:
            # `closed_channels` is deliberately left empty. The Smart Agent passes the
            # channels the Circuit Breaker actually reported closed, because that is the
            # honest "because" behind its escalation. This agent closed nothing — the
            # channel is still perfectly usable and it simply ran out of counter — so
            # naming channels here would be claiming a reason it never computed
            # (see `human_fallback.run_human_fallback`, which allows exactly this).
            result.human_fallback_event = run_human_fallback(transaction, now, merchant)
            result.terminal_reason = TERMINAL_ATTEMPT_CAP_REACHED
            break

        attempt_at = next_attempt_at(now, config)

        if attempt_at > horizon:
            transaction.status = TransactionStatus.ABANDONED
            result.terminal_reason = TERMINAL_HORIZON_EXCEEDED
            break

        clock.advance(attempt_at - now)

        # `route` stays None throughout: route-level arms are not modelled anywhere in
        # this build, so "same channel, same route" reduces to "same channel" here
        # exactly as it does for the Smart Agent.
        outcome, decline_code, _simulated_category = simulator.respond_to_attempt(
            transaction, channel, None, attempt_at
        )
        success = outcome == AttemptOutcome.APPROVED

        transaction.record_attempt(
            AttemptRecord(
                transaction_id=transaction.transaction_id,
                attempt_number=len(transaction.attempts) + 1,
                channel=channel,
                route=None,
                attempted_at=attempt_at,
                chosen_arm=BASELINE_ARM,
                outcome=outcome,
                decline_code=decline_code,
                # Left as None on purpose, and it is the most important `None` in this
                # file. The bank's code is a fact this agent received, so it is written
                # down; the soft/hard category is an *interpretation* it never performed.
                # Filling it in — even correctly, from the registry — would fabricate a
                # diagnosis step this agent does not have, and would corrupt Phase 11's
                # classifier-accuracy metric, which is a Smart-Agent-only measurement.
                decline_category=None,
            )
        )

        result.decisions.append(
            BaselineAttemptDecision(
                transaction_id=transaction.transaction_id,
                attempt_number=len(transaction.attempts),
                decided_at=now,
                chosen_channel=channel,
                chosen_arm=BASELINE_ARM,
                wait_hours=wait_hours(config),
                attempt_cap=config.max_attempts,
                attempted_at=attempt_at,
                outcome=outcome,
                decline_code=decline_code,
            )
        )

        if success:
            transaction.status = TransactionStatus.RECOVERED
            result.terminal_reason = TERMINAL_RECOVERED
            break

    return result


def iter_batch(
    simulator: Simulator,
    config: Optional[BaselineConfig] = None,
) -> Iterator[BaselineTransactionResult]:
    """Run the Baseline Agent over every failed billing event in a seeded Simulator
    batch, yielding each transaction as it finishes (ARCHITECTURE.md C2; the mirror of
    `orchestrator.iter_batch`).

    Billing events that never failed are skipped, for the same reason the Orchestrator
    skips them: there is nothing to recover, and manufacturing a transaction for one
    would inflate the recovery rate with a case that was never at risk. Both agents must
    apply that filter identically or the denominators stop matching.

    Note the signature against `orchestrator.run_batch`, which threads a
    `SmartAgentRuntime` through the entire batch. There is no equivalent here, and that
    missing parameter is the most compact statement of what this agent is. The Smart
    Agent carries two things from one transaction into the next: the Bandit's pool of
    evidence (so what it learns from customer 1 helps customer 900) and Pacing's rolling
    view of system health (so a wave of failures slows everything down). This agent
    carries nothing. Each transaction is handled as though it were the first and only one
    ever seen, and the batch is genuinely just a loop.
    """
    config = config or BaselineConfig()

    customers_by_id = {c.customer_id: c for c in simulator.customers}
    merchants_by_id = {m.merchant_id: m for m in simulator.merchants}

    for billing_event in simulator.billing_events:
        if billing_event.billing_event_id not in simulator.failed_billing_event_ids:
            continue
        yield run_transaction(
            simulator,
            billing_event,
            customers_by_id[billing_event.customer_id],
            merchants_by_id[billing_event.merchant_id],
            config,
        )


def run_batch(
    simulator: Simulator,
    config: Optional[BaselineConfig] = None,
) -> list[BaselineTransactionResult]:
    """The whole Baseline batch, run to completion — `iter_batch` drained into a list.
    Unchanged in behaviour and signature; the mirror of `orchestrator.run_batch`."""
    return list(iter_batch(simulator, config))


AgentResult = Union[TransactionResult, BaselineTransactionResult]
"""Either agent's per-transaction result — the two things a paired run produces.

Defined here rather than in `audit_trail` (which is where it was first needed, and which
re-exports it) because this is the lowest module that knows about both halves: it defines
`BaselineTransactionResult` and imports `TransactionResult`. One definition, so a third
agent could never be added to one copy and not the other.
"""


@dataclass(frozen=True)
class PairedRun:
    """Both agents' output over one identical seeded world — the object the whole
    project exists to produce.

    This is what ARCHITECTURE.md C2 operation 5 ("must run against **exactly** the same
    seeded Simulator population as the Smart Agent") actually amounts to in code, and
    what makes Phase 11's comparison a controlled experiment rather than two unrelated
    numbers printed near each other.

    `simulator` is exposed for its **population and Hidden Truth records**, which the
    Metrics engine (C3) needs privately to answer the against-truth questions. Those are
    fixed at construction time and identical across both agents' worlds by definition.
    Its RNG position, by contrast, is meaningless once a run has finished — do not use
    this object to make further attempts and expect them to mean anything.
    """

    simulator_config: SimulatorConfig
    simulator: Simulator
    smart: list[TransactionResult]
    baseline: list[BaselineTransactionResult]

    orchestrator_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    baseline_config: "BaselineConfig" = field(default_factory=lambda: BaselineConfig())
    """The *effective* agent configurations these results were produced with, resolved
    to their defaults rather than left as `None`.

    Carried on the run for the same reason `simulator_config` already is: the Metrics
    engine (C3, operation 8) has to key a results object by everything that determined
    the outcome, and a comparison stamped with a configuration supplied separately by
    the caller is a citation nobody should trust. A seed alone does not identify a run
    either — the same seed at a different failure rate is a different world.
    """


def iter_paired_batch(
    simulator_config: SimulatorConfig,
    orchestrator_config: Optional[OrchestratorConfig] = None,
    baseline_config: Optional[BaselineConfig] = None,
) -> Generator[AgentResult, None, PairedRun]:
    """Run both agents over the same seeded world, yielding each transaction as it
    finishes and *returning* the assembled `PairedRun` when the run is over
    (ARCHITECTURE.md C2, operation 5; D1, operation 12).

    **The two-value shape, and why it is one function rather than two.** The Dashboard
    (C4) needs both halves of the same run: each finished transaction while the batch is
    still going, to narrate live (operation 1), and the complete `PairedRun` at the end,
    to hand to the Metrics engine (operations 2-3). Splitting those into a streaming
    function and a separate re-run would run the world twice and invite the two copies to
    disagree. A generator's return value carries the finished object out of the same
    single pass — a consumer that only wants the end state calls `run_paired_batch`
    below and never sees the stream at all.

    Yielded results are *not* labelled with which agent produced them, deliberately:
    `audit_trail.agent_for` already reads that off the result's type, and a second copy of
    that mapping living here is exactly the kind of duplicate that drifts. (This module
    cannot import `audit_trail` in any case — that module imports this one.)

    Everything that decides what the world *is* — which customers exist, which mandates
    they hold, which billing events fail, what the first decline code was, and the entire
    Hidden Truth answer key — is generated once in `Simulator.__init__` from
    `simulator_config.seed`. So two Simulators built from the same config are the same
    world, down to the last customer.

    **Why each agent gets its own Simulator instance rather than sharing one.** Sharing
    would be *nearly* fine, and it is worth being precise about the "nearly" rather than
    hand-waving it. Approve/decline verdicts are not random at attempt time: they are a
    pure function of the channel, the elapsed simulated time and the attempt number,
    checked against a Hidden Truth that was fixed at construction. What *is* drawn from
    the Simulator's RNG during a run is the choice of *which* decline code to show within
    an already-decided category. So a shared instance would leave whichever agent ran
    second reading from a different RNG position and seeing different code strings for
    identical verdicts — cosmetic, not a fairness bug, but visible in the audit trail and
    tedious to have to explain.

    Building a second Simulator costs a few milliseconds and removes the question
    entirely, which is the right trade when the requirement being satisfied is literally
    the word "identical". The alternative — threading a separate RNG through
    `simulator.respond_to_attempt` — would mean reopening a finished Phase 2 component
    for a cosmetic gain, so it was not done.

    Note the deliberate ordering-independence this buys: neither agent can affect the
    other's run, so `smart` and `baseline` could be computed in either order, or years
    apart, and be identical either way.
    """
    # Resolved here rather than left to each agent's own `config or Default()` so that
    # the run records the configuration it actually ran under, not the argument it was
    # handed. Behaviourally identical; the difference is that the result can now say
    # what produced it.
    orchestrator_config = orchestrator_config or OrchestratorConfig()
    baseline_config = baseline_config or BaselineConfig()

    smart_simulator = Simulator(simulator_config)
    baseline_simulator = Simulator(simulator_config)

    smart: list[TransactionResult] = []
    for result in iter_smart_batch(smart_simulator, config=orchestrator_config):
        smart.append(result)
        yield result

    baseline: list[BaselineTransactionResult] = []
    for result in iter_batch(baseline_simulator, config=baseline_config):
        baseline.append(result)
        yield result

    return PairedRun(
        simulator_config=simulator_config,
        simulator=smart_simulator,
        smart=smart,
        baseline=baseline,
        orchestrator_config=orchestrator_config,
        baseline_config=baseline_config,
    )


def run_paired_batch(
    simulator_config: SimulatorConfig,
    orchestrator_config: Optional[OrchestratorConfig] = None,
    baseline_config: Optional[BaselineConfig] = None,
) -> PairedRun:
    """Both agents' run over one identical seeded world, run to completion.

    The form every phase since Phase 9 has called, and still the one to call unless you
    specifically want to watch the batch happen. `iter_paired_batch` above holds the
    reasoning about how the world is built; this drains it and hands back the object it
    returns.

    The loop below is the standard way to reach a generator's `return` value: iterating
    it to exhaustion raises `StopIteration`, whose `value` is what the generator
    returned. Results are dropped as they arrive because the generator is accumulating
    them into the `PairedRun` regardless.
    """
    stream = iter_paired_batch(
        simulator_config,
        orchestrator_config=orchestrator_config,
        baseline_config=baseline_config,
    )
    while True:
        try:
            next(stream)
        except StopIteration as finished:
            return finished.value
