"""A1 — Data Model / Entity Registry.

The shared vocabulary every other component (simulator, classifier, bandit, circuit
breaker, orchestrator, ...) reads and writes. These are plain typed/validated record
shapes, not components that make decisions on their own — see notes/ARCHITECTURE.md
Part A1 for the full role description.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

_DECLINE_CODES_DIR = Path(__file__).resolve().parent.parent / "data" / "decline_codes"


class MandateChannel(str, Enum):
    """The three kinds of standing mandate a customer can have on file."""

    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"


class DeclineCategory(str, Enum):
    """The one soft-vs-hard concept every channel's decline codes get sorted into."""

    SOFT = "soft"
    HARD = "hard"


class AttemptOutcome(str, Enum):
    """What the (simulated) bank said back for a single retry attempt."""

    APPROVED = "approved"
    DECLINED = "declined"


class TransactionStatus(str, Enum):
    """The lifecycle states of one billing event's recovery journey."""

    IN_PROGRESS = "in_progress"
    RECOVERED = "recovered"
    ABANDONED = "abandoned"
    ESCALATED_TO_HUMAN = "escalated_to_human"


class Merchant(BaseModel):
    """An identity a customer's mandate is registered against (e.g. a subscription
    business like a streaming service or a gym)."""

    merchant_id: str
    display_name: str


class MandateRegistration(BaseModel):
    """One channel-on-file for one customer — e.g. "customer #142 has a card mandate
    registered." A customer can have zero or more of these, one per channel."""

    channel: MandateChannel
    registered_at: datetime


class Customer(BaseModel):
    """A person with one active subscription against one merchant.

    Holds which mandate channel(s) they have registered (a non-empty subset of
    card/UPI/netbanking) and their billing cycle length.
    """

    customer_id: str
    merchant_id: str
    mandates: list[MandateRegistration] = Field(min_length=1)
    billing_cycle_days: int = 30

    def registered_channels(self) -> set[MandateChannel]:
        """The set of channels this customer has on file — for membership tests only.

        A `set` has no stable iteration order across processes (Python randomizes string
        hashing per run), so anything that *iterates* this — especially anything that
        then draws from the result with a seeded RNG — silently loses seeded
        reproducibility. Use `ordered_channels()` for that. See its docstring for the
        bug this cost us.
        """
        return {m.channel for m in self.mandates}

    def ordered_channels(self) -> list[MandateChannel]:
        """This customer's channels in registration order (primary first), deduplicated.

        The deterministic counterpart to `registered_channels()`, and the one every
        caller that iterates, samples or chooses should use.

        This exists because iterating the set version cost us a real bug, found during
        Phase 10: the Simulator built each failed event's list of alternative channels by
        iterating `registered_channels()`, then drew the genuinely-recoverable one from
        that list with its seeded RNG. The draw was seeded and reproducible; the *order
        it drew from* was not. So the Hidden Truth answer key — which channel would have
        worked — differed between two runs of the same seed in different processes, and
        with it the Smart Agent's recovery numbers. Within a single process both agents
        still saw one identical world, which is why the Phase 9 comparison stayed valid
        and why the problem stayed invisible until an audit trail was diffed across runs.
        """
        seen: set[MandateChannel] = set()
        ordered: list[MandateChannel] = []
        for mandate in self.mandates:
            if mandate.channel not in seen:
                seen.add(mandate.channel)
                ordered.append(mandate.channel)
        return ordered

    def primary_channel(self) -> MandateChannel:
        """The customer's first-registered channel — always the one charged on a
        billing event's first attempt (per ARCHITECTURE.md A2, operation 5)."""
        return self.mandates[0].channel


class BillingEvent(BaseModel):
    """One occurrence of "this customer's subscription came due."""

    billing_event_id: str
    customer_id: str
    merchant_id: str
    scheduled_at: datetime


class HiddenTruthRecord(BaseModel):
    """The answer key attached privately to a billing event when it fails.

    Generated once by the Simulator and then fixed — never re-rolled per attempt. Not
    visible to any decision-making component; only to the Simulator's own response
    function and to the Metrics engine afterward (ARCHITECTURE.md A1/A2).
    """

    billing_event_id: str
    is_recoverable: bool
    recoverable_channels: list[MandateChannel] = Field(default_factory=list)
    recoverable_after_seconds: Optional[float] = None
    recoverable_on_attempt_number: Optional[int] = None


class AttemptRecord(BaseModel):
    """One single retry attempt against a transaction."""

    transaction_id: str
    attempt_number: int
    channel: MandateChannel
    route: Optional[str] = None
    attempted_at: datetime
    chosen_arm: str
    outcome: AttemptOutcome
    decline_code: Optional[str] = None
    decline_category: Optional[DeclineCategory] = None


class TransactionState(BaseModel):
    """The running record for one billing event's whole recovery journey."""

    transaction_id: str
    billing_event_id: str
    customer_id: str
    merchant_id: str
    status: TransactionStatus = TransactionStatus.IN_PROGRESS
    attempts: list[AttemptRecord] = Field(default_factory=list)

    def attempts_for_channel(self, channel: MandateChannel) -> list[AttemptRecord]:
        return [a for a in self.attempts if a.channel == channel]

    def channel_attempt_count(self, channel: MandateChannel) -> int:
        return len(self.attempts_for_channel(channel))

    def record_attempt(self, attempt: AttemptRecord) -> None:
        self.attempts.append(attempt)


class DeclineCodeRegistry:
    """Read-only, static lookup access to the three per-channel decline code tables
    in `data/decline_codes/`. Loaded once per channel and cached — this is reference
    data, not something regenerated per run (ARCHITECTURE.md A1, operation 3).
    """

    _cache: dict[MandateChannel, dict[str, DeclineCategory]] = {}
    _raw_cache: dict[MandateChannel, dict[str, dict[str, str]]] = {}

    @classmethod
    def _load_raw(cls, channel: MandateChannel) -> dict[str, dict[str, str]]:
        if channel not in cls._raw_cache:
            path = _DECLINE_CODES_DIR / f"{channel.value}.json"
            cls._raw_cache[channel] = json.loads(path.read_text(encoding="utf-8"))["codes"]
        return cls._raw_cache[channel]

    @classmethod
    def _load_channel(cls, channel: MandateChannel) -> dict[str, DeclineCategory]:
        if channel not in cls._cache:
            cls._cache[channel] = {
                code: DeclineCategory(entry["category"])
                for code, entry in cls._load_raw(channel).items()
            }
        return cls._cache[channel]

    @classmethod
    def lookup(cls, channel: MandateChannel, code: str) -> Optional[DeclineCategory]:
        """Return the category for a (channel, code) pair, or None if it isn't a
        recognized code — callers (e.g. the Classifier, B1) must handle None by
        flagging the pair explicitly rather than guessing a category."""
        return cls._load_channel(channel).get(code)

    @classmethod
    def describe(cls, channel: MandateChannel, code: str) -> Optional[str]:
        """The registry's own short human description of a code (e.g. `"51"` ->
        `"Insufficient funds"`), or None if the pair isn't recognized.

        Added for the Audit Trail (C1), which has to write a sentence a non-engineer can
        read: "declined with code 51 (insufficient funds)" is the same fact as "declined
        with code 51", stated in language a merchant understands. It is a *lookup in
        reference data*, not an interpretation — exactly like `lookup` — which is why it
        belongs here in A1 rather than as a table of prose inside C1. Returning None for
        an unknown code keeps the same honesty contract: the caller flags it, nobody
        guesses.
        """
        entry = cls._load_raw(channel).get(code)
        return entry.get("description") if entry else None

    @classmethod
    def is_known(cls, channel: MandateChannel, code: str) -> bool:
        return code in cls._load_channel(channel)

    @classmethod
    def all_codes(cls, channel: MandateChannel) -> dict[str, DeclineCategory]:
        return dict(cls._load_channel(channel))
