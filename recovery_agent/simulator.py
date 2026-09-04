"""A2 — Simulator Engine.

The fake bank in a box. Generates a realistic population of customers and billing
failures, and (once respond_to_attempt is built, later in this phase) answers "did this
attempt succeed?" honestly according to a hidden schedule only it knows. See
notes/ARCHITECTURE.md Part A2 for the full role description.

Built incrementally across Phase 2's task list (notes/PLAN.md) — this file currently
covers: merchant/customer generation with a configurable mandate-channel-profile mix,
billing event generation per customer per cycle with a configurable base failure rate,
failure category + specific decline code assignment on each failed event's first
attempt, and Hidden Truth Record generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from recovery_agent.models import (
    AttemptOutcome,
    BillingEvent,
    Customer,
    DeclineCategory,
    DeclineCodeRegistry,
    HiddenTruthRecord,
    MandateChannel,
    MandateRegistration,
    Merchant,
    TransactionState,
)

# Default demo merchant flavor — a small fixed set of fake identities, per
# ARCHITECTURE.md A2 operation 1 ("a small fixed set of fake merchant identities").
DEFAULT_MERCHANT_NAMES: list[str] = [
    "StreamFlix",
    "FitCore Gym",
    "CloudNotes Pro",
    "DailyBrew Subscription",
    "SkillForge Academy",
]

# Default mandate-channel-profile mix, per DESIGN.md/ARCHITECTURE.md's own example:
# 70% card-only, 20% card+UPI, 10% all three. Channel order within each tuple matters —
# the first channel listed is the customer's primary channel (Customer.primary_channel()
# in models.py), always the one charged on a billing event's first attempt.
DEFAULT_MANDATE_PROFILE_MIX: dict[tuple[MandateChannel, ...], float] = {
    (MandateChannel.CARD,): 0.70,
    (MandateChannel.CARD, MandateChannel.UPI): 0.20,
    (MandateChannel.CARD, MandateChannel.UPI, MandateChannel.NETBANKING): 0.10,
}


@dataclass
class SimulatorConfig:
    """Configuration only — per ARCHITECTURE.md A2, the Simulator's sole input."""

    seed: int
    num_customers: int = 200
    merchant_names: list[str] = field(default_factory=lambda: list(DEFAULT_MERCHANT_NAMES))
    mandate_profile_mix: dict[tuple[MandateChannel, ...], float] = field(
        default_factory=lambda: dict(DEFAULT_MANDATE_PROFILE_MIX)
    )
    billing_cycle_days: int = 30
    start_time: datetime = field(default_factory=lambda: datetime(2026, 1, 1))
    num_cycles: int = 1
    # ~8% is a commonly-cited ballpark industry figure for subscription payment failure
    # rates (not tied to any one specific channel or provider) — a reasonable default to
    # calibrate the simulator with, adjustable per run.
    base_failure_rate: float = 0.08
    # Most declines are soft (temporary), a minority hard (permanent) — per
    # ARCHITECTURE.md A2 operation 4 ("weighted realistically").
    soft_failure_ratio: float = 0.8
    # Hidden Truth generation — how likely a failure is genuinely recoverable, and
    # roughly how long/how many attempts it takes if the agent behaves optimally. Our
    # own reasonable modeling assumptions (soft declines are only "maybe" recoverable
    # per notes/IDEA.md §7, not guaranteed even on the same channel), not derived from
    # any published recovery-rate source.
    soft_recoverable_ratio: float = 0.85
    hard_recoverable_via_switch_ratio: float = 0.5
    recoverable_wait_hours_range: tuple[float, float] = (1.0, 48.0)
    # "Mass failure" scenario mode, per ARCHITECTURE.md A2 operation 10 — force a
    # fraction of customers' billing events to fail at the *same* simulated instant,
    # specifically to give the Pacing component's Jitter half (B3, Phase 6) a
    # thundering-herd condition to be tested against. Off by default; a normal batch
    # run spreads each customer's billing event across the cycle, on their own renewal
    # anniversary (`Customer.billing_anniversary_offset_days`), so this mode genuinely
    # changes *when* failures land and not merely how many there are. Until Phase 13
    # that contrast was fiction — every customer shared one renewal instant, so the
    # herd was always there and this toggle only made it bigger.
    mass_failure_scenario: bool = False
    mass_failure_fraction: float = 0.3


class SimulatedClock:
    """A simulated clock that can be advanced arbitrarily (simulated hours/days) with
    no real waiting — per ARCHITECTURE.md A2 operation 7. Nothing here reads the real
    system clock; "now" is purely whatever this object says it is, so a whole batch of
    thousands of retry waits can be fast-forwarded through instantly."""

    def __init__(self, start_time: datetime):
        self.current_time = start_time

    def now(self) -> datetime:
        return self.current_time

    def advance(self, delta: timedelta) -> datetime:
        self.current_time += delta
        return self.current_time

    def advance_hours(self, hours: float) -> datetime:
        return self.advance(timedelta(hours=hours))


class Simulator:
    """The fake bank in a box. Holds its own seeded RNG so that two Simulators built
    from the same SimulatorConfig.seed always produce identical output (A2 operation 9)."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.merchants: list[Merchant] = []
        self.customers: list[Customer] = []
        self.billing_events: list[BillingEvent] = []
        # Which billing events rolled a failure on generation — checked here once,
        # rather than re-rolled later, so a given event's fail/succeed fate is fixed
        # from the moment it's created (the Hidden Truth Record, built in a later task,
        # attaches its recoverability detail only to events found in this set).
        self.failed_billing_event_ids: set[str] = set()
        # The first attempt's diagnosis for each failed event, keyed by billing_event_id
        # — always rolled on the customer's primary channel (models.py,
        # Customer.primary_channel()), per ARCHITECTURE.md A2 operation 5.
        self.first_attempt_category: dict[str, DeclineCategory] = {}
        self.first_attempt_code: dict[str, str] = {}
        # The private "answer key" for each failed billing event — see
        # models.HiddenTruthRecord. Only this class's own respond_to_attempt (later
        # task) and the Metrics engine (Phase 11) are meant to ever read this.
        self.hidden_truths: dict[str, HiddenTruthRecord] = {}
        self._customers_by_id: dict[str, Customer] = {}
        self._billing_events_by_id: dict[str, BillingEvent] = {}
        # The simulated clock (A2 operation 7) — starts at the same instant the
        # population's billing events are scheduled from, so "elapsed time since
        # failure" in respond_to_attempt has a sane zero point.
        self.clock = SimulatedClock(config.start_time)
        self._generate_merchants()
        self._generate_customers()
        self._generate_billing_events()
        self._assign_first_attempt_declines()
        self._generate_hidden_truths()

    def _generate_merchants(self) -> None:
        """ARCHITECTURE.md A2 operation 1 — a small fixed set of fake merchant
        identities, one per configured name."""
        self.merchants = [
            Merchant(merchant_id=f"merchant_{i + 1}", display_name=name)
            for i, name in enumerate(self.config.merchant_names)
        ]

    def _generate_customers(self) -> None:
        """ARCHITECTURE.md A2 operation 2 — assign each customer a merchant, a
        mandate-channel profile (drawn from the configured mix), and a billing cycle."""
        profiles = list(self.config.mandate_profile_mix.keys())
        weights = list(self.config.mandate_profile_mix.values())

        for i in range(self.config.num_customers):
            merchant = self.rng.choice(self.merchants)
            channels = self.rng.choices(profiles, weights=weights, k=1)[0]
            mandates = [
                MandateRegistration(channel=channel, registered_at=self.config.start_time)
                for channel in channels
            ]
            # Each customer renews on their own anniversary within the cycle, not on a
            # date the whole population shares — see
            # `Customer.billing_anniversary_offset_days`.
            offset_days = self.rng.randrange(self.config.billing_cycle_days)
            customer = Customer(
                customer_id=f"customer_{i + 1}",
                merchant_id=merchant.merchant_id,
                mandates=mandates,
                billing_cycle_days=self.config.billing_cycle_days,
                billing_anniversary_offset_days=offset_days,
            )
            self.customers.append(customer)

        self._customers_by_id = {c.customer_id: c for c in self.customers}

    def _generate_billing_events(self) -> None:
        """ARCHITECTURE.md A2 operation 3 — one billing event per customer per
        configured cycle, each rolling independently whether it fails.

        Normally each customer is charged on their own renewal anniversary within the
        cycle (`Customer.billing_anniversary_offset_days`), so a batch's charges are
        spread across the whole cycle rather than piled onto one instant.

        If `mass_failure_scenario` is enabled (operation 10), a configured fraction of
        customers are pre-selected (via the same seeded RNG, so it's still
        reproducible) to have their *first* cycle's billing event forced to fail and
        forced onto the exact same simulated instant, overriding their anniversary — a
        thundering-herd condition for exercising the Pacing component's Jitter half
        (Phase 6) against the spread-out normal batch."""
        mass_failure_customer_ids: set[str] = set()
        # Mid-cycle: the outage lands in the middle of the window renewals are spread
        # across, so the forced herd sits inside a normal batch rather than at its edge.
        mass_failure_instant = self.config.start_time + timedelta(
            days=self.config.billing_cycle_days * 1.5
        )
        if self.config.mass_failure_scenario:
            num_mass_failures = round(len(self.customers) * self.config.mass_failure_fraction)
            mass_failure_customer_ids = {
                c.customer_id
                for c in self.rng.sample(self.customers, min(num_mass_failures, len(self.customers)))
            }

        for customer in self.customers:
            for cycle_index in range(self.config.num_cycles):
                event_id = f"{customer.customer_id}_cycle{cycle_index + 1}"
                is_mass_failure = cycle_index == 0 and customer.customer_id in mass_failure_customer_ids
                scheduled_at = (
                    mass_failure_instant
                    if is_mass_failure
                    else self.config.start_time
                    + timedelta(
                        days=customer.billing_cycle_days * (cycle_index + 1)
                        + customer.billing_anniversary_offset_days
                    )
                )
                event = BillingEvent(
                    billing_event_id=event_id,
                    customer_id=customer.customer_id,
                    merchant_id=customer.merchant_id,
                    scheduled_at=scheduled_at,
                )
                self.billing_events.append(event)
                self._billing_events_by_id[event_id] = event
                if is_mass_failure or self.rng.random() < self.config.base_failure_rate:
                    self.failed_billing_event_ids.add(event_id)

        # Chronological order, tie-broken by id so it stays deterministic. Both agents
        # walk this one list (`orchestrator.iter_batch`, `baseline_agent.iter_batch`), so
        # ordering it here orders both identically and keeps the comparison fair.
        # It matters now that anniversaries are staggered: in customer-generation order a
        # transaction due on day 59 would be processed before one due on day 31, which
        # would make Pacing's "recent outcomes" window read the batch in an order the
        # simulated clock never had. This does not make the window time-ordered — one
        # transaction still runs all its retries before the next begins — it only stops
        # the staggering from making that gap worse. See notes/TRACKER.md's
        # "system-wide is processing-order" limitation.
        self.billing_events.sort(key=lambda e: (e.scheduled_at, e.billing_event_id))

    def _assign_first_attempt_declines(self) -> None:
        """ARCHITECTURE.md A2 operations 4-5 — for each failed billing event, assign a
        soft/hard category, then a specific decline code within that category, on the
        customer's primary (first-registered) channel."""
        for event in self.billing_events:
            if event.billing_event_id not in self.failed_billing_event_ids:
                continue
            customer = self._customers_by_id[event.customer_id]
            channel = customer.primary_channel()

            category = (
                DeclineCategory.SOFT
                if self.rng.random() < self.config.soft_failure_ratio
                else DeclineCategory.HARD
            )
            codes_in_category = [
                code
                for code, cat in DeclineCodeRegistry.all_codes(channel).items()
                if cat == category
            ]
            code = self.rng.choice(codes_in_category)

            self.first_attempt_category[event.billing_event_id] = category
            self.first_attempt_code[event.billing_event_id] = code

    def _generate_hidden_truths(self) -> None:
        """ARCHITECTURE.md A2 operation 6 — for each failed billing event, decide once
        (and fix permanently) whether it's genuinely recoverable, through which
        channel(s), after roughly how long, and on which attempt number it would first
        succeed if the agent behaves optimally. Never re-rolled later — no
        decision-making component ever sees this, only this class's own
        respond_to_attempt (a later task) and the Metrics engine (Phase 11)."""
        for event in self.billing_events:
            if event.billing_event_id not in self.failed_billing_event_ids:
                continue
            customer = self._customers_by_id[event.customer_id]
            category = self.first_attempt_category[event.billing_event_id]
            primary = customer.primary_channel()
            # Registration order, never `registered_channels()` — this list is drawn
            # from by the seeded RNG below, and a set's iteration order is not stable
            # across processes, which silently unseeded the Hidden Truth (see
            # `Customer.ordered_channels`).
            other_channels = [c for c in customer.ordered_channels() if c != primary]

            if category == DeclineCategory.HARD:
                # A hard decline means the primary channel itself is permanently
                # dead — recovery is only possible by switching to another
                # registered channel, and only sometimes even then.
                is_recoverable = bool(other_channels) and (
                    self.rng.random() < self.config.hard_recoverable_via_switch_ratio
                )
                recoverable_channels = (
                    [self.rng.choice(other_channels)] if is_recoverable else []
                )
            else:
                # A soft decline is only "maybe" recoverable (notes/IDEA.md §7) — not
                # guaranteed, even on the same channel.
                is_recoverable = self.rng.random() < self.config.soft_recoverable_ratio
                if is_recoverable:
                    candidates = [primary, *other_channels]
                    num_channels = 2 if other_channels and self.rng.random() < 0.2 else 1
                    recoverable_channels = self.rng.sample(candidates, num_channels)
                else:
                    recoverable_channels = []

            if not is_recoverable:
                record = HiddenTruthRecord(
                    billing_event_id=event.billing_event_id,
                    is_recoverable=False,
                )
            else:
                wait_hours = self.rng.uniform(*self.config.recoverable_wait_hours_range)
                attempt_number = self.rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
                record = HiddenTruthRecord(
                    billing_event_id=event.billing_event_id,
                    is_recoverable=True,
                    recoverable_channels=recoverable_channels,
                    recoverable_after_seconds=wait_hours * 3600,
                    recoverable_on_attempt_number=attempt_number,
                )

            self.hidden_truths[event.billing_event_id] = record

    def respond_to_attempt(
        self,
        transaction: TransactionState,
        channel: MandateChannel,
        route: Optional[str],
        current_time: datetime,
    ) -> tuple[AttemptOutcome, Optional[str], Optional[DeclineCategory]]:
        """ARCHITECTURE.md A2 operation 8 — the core "fake bank" verdict function.

        Given a transaction, the channel/route an attempt is being made on, and the
        current simulated time, consults that transaction's Hidden Truth Record (never
        exposed to any decision-making component) and the simulated time elapsed since
        the original failure, and returns approve, or decline + a specific decline code
        and its category.

        `route` is accepted per the architecture's function signature (a card-specific
        sub-arm, e.g. a different acquiring route for the same card) but the current
        Hidden Truth model doesn't distinguish outcomes by route within a channel — only
        by channel. Recorded for future use, not consulted yet.
        """
        del route  # not yet modeled — see docstring
        truth = self.hidden_truths.get(transaction.billing_event_id)
        if truth is None:
            # No failure was ever generated for this billing event, so there's nothing
            # to recover from — treat any attempt against it as trivially approved.
            return AttemptOutcome.APPROVED, None, None

        billing_event = self._billing_events_by_id[transaction.billing_event_id]
        customer = self._customers_by_id[transaction.customer_id]
        # Retry number counts attempts already recorded on this transaction's own
        # journey (i.e. retries after the original failure), matching how
        # `recoverable_on_attempt_number` was generated in _generate_hidden_truths.
        retry_number = len(transaction.attempts) + 1
        elapsed_seconds = (current_time - billing_event.scheduled_at).total_seconds()

        channel_is_recoverable = truth.is_recoverable and channel in truth.recoverable_channels
        if (
            channel_is_recoverable
            and elapsed_seconds >= (truth.recoverable_after_seconds or 0.0)
            and retry_number >= (truth.recoverable_on_attempt_number or 1)
        ):
            return AttemptOutcome.APPROVED, None, None

        # Still declined. Pick a category honestly reflecting the truth we're
        # withholding: a channel that *will* eventually work just hasn't met its wait
        # time / attempt-number condition yet, so it reads as a soft (temporary)
        # decline; a channel with no path to recovery at all reads as hard only when
        # it's the customer's primary channel and the original diagnosis was hard
        # (a permanently dead primary channel stays hard on every retry) — every other
        # never-recoverable case still reads as soft, since the agent has no way of
        # knowing in advance that persistence here is futile.
        if channel_is_recoverable:
            category = DeclineCategory.SOFT
        elif (
            channel == customer.primary_channel()
            and self.first_attempt_category.get(transaction.billing_event_id) == DeclineCategory.HARD
        ):
            category = DeclineCategory.HARD
        else:
            category = DeclineCategory.SOFT

        codes_in_category = [
            code
            for code, cat in DeclineCodeRegistry.all_codes(channel).items()
            if cat == category
        ]
        code = self.rng.choice(codes_in_category)
        return AttemptOutcome.DECLINED, code, category
