"""B1 — Classifier.

The diagnosis step: turns a raw (channel, decline code) pair into a soft/hard
category by looking it up in the Decline Code Registry (A1). See
notes/ARCHITECTURE.md Part B1 for the full role description.
"""

from __future__ import annotations

from recovery_agent.models import DeclineCategory, DeclineCodeRegistry, MandateChannel


class UnrecognizedDeclineCodeError(ValueError):
    """Raised when a (channel, code) pair isn't in the Decline Code Registry.

    Per ARCHITECTURE.md B1 operation 3, an unrecognized pair must be flagged
    explicitly rather than silently guessed a category — this is the honesty
    requirement carried over from A1.
    """

    def __init__(self, channel: MandateChannel, code: str) -> None:
        self.channel = channel
        self.code = code
        super().__init__(
            f"Unrecognized decline code {code!r} for channel {channel.value!r} — "
            "refusing to guess a soft/hard category."
        )


def classify(channel: MandateChannel, code: str) -> DeclineCategory:
    """Return the soft/hard category for a (channel, code) pair.

    Raises UnrecognizedDeclineCodeError if the pair isn't in the registry,
    rather than guessing.
    """
    category = DeclineCodeRegistry.lookup(channel, code)
    if category is None:
        raise UnrecognizedDeclineCodeError(channel, code)
    return category
