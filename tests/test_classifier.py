"""Tests for B1 — Classifier (recovery_agent/classifier.py).

Per notes/PLAN.md's Phase 3 verification step: known soft codes -> soft, known
hard codes -> hard, unknown code -> flagged/raised, across all three channels.
"""

import pytest

from recovery_agent.classifier import UnrecognizedDeclineCodeError, classify
from recovery_agent.models import DeclineCategory, DeclineCodeRegistry, MandateChannel


@pytest.mark.parametrize(
    "channel,code",
    [
        (MandateChannel.CARD, "51"),  # Insufficient funds
        (MandateChannel.CARD, "91"),  # Issuer timeout
        (MandateChannel.UPI, "PIN_ENTRY_TIMEOUT"),
        (MandateChannel.UPI, "BANK_SERVER_UNREACHABLE"),
        (MandateChannel.NETBANKING, "SESSION_TIMEOUT"),
        (MandateChannel.NETBANKING, "BANK_PORTAL_DOWN"),
    ],
)
def test_known_soft_codes_classify_as_soft(channel, code):
    assert classify(channel, code) == DeclineCategory.SOFT


@pytest.mark.parametrize(
    "channel,code",
    [
        (MandateChannel.CARD, "54"),  # Expired card
        (MandateChannel.CARD, "41"),  # Stolen card, pick up
        (MandateChannel.UPI, "MANDATE_REVOKED"),
        (MandateChannel.UPI, "INVALID_VPA"),
        (MandateChannel.NETBANKING, "SI_CANCELLED"),
        (MandateChannel.NETBANKING, "ACCOUNT_CLOSED"),
    ],
)
def test_known_hard_codes_classify_as_hard(channel, code):
    assert classify(channel, code) == DeclineCategory.HARD


@pytest.mark.parametrize(
    "channel",
    [MandateChannel.CARD, MandateChannel.UPI, MandateChannel.NETBANKING],
)
def test_unrecognized_code_is_flagged_not_guessed(channel):
    with pytest.raises(UnrecognizedDeclineCodeError) as exc_info:
        classify(channel, "NOT_A_REAL_CODE")
    assert exc_info.value.channel == channel
    assert exc_info.value.code == "NOT_A_REAL_CODE"


@pytest.mark.parametrize(
    "channel",
    [MandateChannel.CARD, MandateChannel.UPI, MandateChannel.NETBANKING],
)
def test_classify_agrees_with_registry_for_every_known_code(channel):
    """Every code in the registry round-trips through classify() unchanged —
    the classifier must never override what the registry itself says."""
    for code, category in DeclineCodeRegistry.all_codes(channel).items():
        assert classify(channel, code) == category
