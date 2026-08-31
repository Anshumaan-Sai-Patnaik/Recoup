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

from recovery_agent.models import (
    BillingEvent,
    Customer,
    DeclineCategory,
    DeclineCodeRegistry,
    HiddenTruthRecord,
    MandateChannel,
    MandateRegistration,
    Merchant,
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
            customer = Customer(
                customer_id=f"customer_{i + 1}",
                merchant_id=merchant.merchant_id,
                mandates=mandates,
                billing_cycle_days=self.config.billing_cycle_days,
            )
            self.customers.append(customer)

        self._customers_by_id = {c.customer_id: c for c in self.customers}

    def _generate_billing_events(self) -> None:
        """ARCHITECTURE.md A2 operation 3 — one billing event per customer per
        configured cycle, each rolling independently whether it fails."""
        for customer in self.customers:
            for cycle_index in range(self.config.num_cycles):
                event_id = f"{customer.customer_id}_cycle{cycle_index + 1}"
                scheduled_at = self.config.start_time + timedelta(
                    days=customer.billing_cycle_days * (cycle_index + 1)
                )
                event = BillingEvent(
                    billing_event_id=event_id,
                    customer_id=customer.customer_id,
                    merchant_id=customer.merchant_id,
                    scheduled_at=scheduled_at,
                )
                self.billing_events.append(event)
                if self.rng.random() < self.config.base_failure_rate:
                    self.failed_billing_event_ids.add(event_id)

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
            other_channels = [c for c in customer.registered_channels() if c != primary]

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
