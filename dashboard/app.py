"""C4 — Dashboard / Reporting Layer (Streamlit entrypoint).

    streamlit run dashboard/app.py

**Presentation only.** ARCHITECTURE.md C4 is explicit that this layer "owns no decision
logic of its own" and is a leaf node: nothing it produces feeds back into any other
component. Every number and every sentence on this page was computed by something else
and is displayed here verbatim. If a fact is not on a result object, this file does not
invent it.

Currently covers C4 operations 1-4. **Operation 1** is the live narration feed, which
IDEA.md §10 names as the single most important live-demo beat: instead of a black box,
judges watch the agent narrate its own reasoning in real time. **Operation 2** is the
head-to-head Baseline-vs-Smart comparison once the batch completes — the "that comparison
*is* the pitch" beat from IDEA.md §11. **Operation 3** marks both agents against the
simulator's private answer key. **Operation 4** hands all of it over as files, so what a
judge saw once in the room can be opened and checked afterwards. **Operation 5** is the
sidebar: which world to build, how the agents are configured, and three presets — one of
which finally switches on the mass-failure scenario the Jitter mechanism exists for.

**Two honesty rules inherited from upstream, restated here because a UI is exactly where
they get broken:**

1. *A blank is a blank.* The Baseline Agent has no Circuit Breaker and no pacing dial, so
   its rows have genuinely empty cells and its metrics have genuinely absent values. A
   chart or table that renders those as `0` would let a reader compare two agents as
   though both had played the same game. Nothing here fills a gap with a plausible
   substitute.
2. *Say only what was recorded.* The narration below is `AuditTrail`'s own rendered
   sentences, unedited. This file adds a timestamp, an agent label and a transaction id —
   all of which the trail already carries — and no prose of its own.

Both rules have teeth: `tests/test_dashboard.py` asserts them against the rendered page,
because a value can be correct in `metrics.py` and still become a false claim on screen.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# `streamlit run dashboard/app.py` executes this file directly rather than importing it
# as part of a package, so the repo root is not automatically on the import path the way
# it is under `pytest` or `python -m`. Adding it here means the documented run command in
# README.md works from a fresh clone with no `PYTHONPATH` ceremony — which is the whole
# deployment story DESIGN.md §6 commits to.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery_agent.audit_trail import (  # noqa: E402
    AGENT_BASELINE,
    AGENT_SMART,
    AuditEntry,
    AuditTrail,
    agent_for,
    reason_phrase,
)
from recovery_agent.baseline_agent import (  # noqa: E402
    AgentResult,
    BaselineConfig,
    PairedRun,
    iter_paired_batch,
)
from recovery_agent import metrics  # noqa: E402
from recovery_agent.metrics import BatchResults  # noqa: E402
from recovery_agent.models import TransactionStatus  # noqa: E402
from recovery_agent.orchestrator import OrchestratorConfig  # noqa: E402
from recovery_agent.simulator import SimulatorConfig  # noqa: E402

# `set_page_config` must be the first Streamlit command a script runs, so it sits at
# module scope rather than inside `main()` — Streamlit raises if any other call beats it.
st.set_page_config(
    page_title="AI Revenue Recovery Agent",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Display vocabulary
#
# The one place this file is allowed an opinion, and it is purely cosmetic: how to spell
# things a person reads. Kept together so no label is invented halfway down the file.
# ---------------------------------------------------------------------------

AGENT_LABELS = {AGENT_SMART: "Smart Agent", AGENT_BASELINE: "Baseline Agent"}

# Chart palette: the first three slots of the project's categorical order, taken in fixed
# order and never cycled. Validated against the light surface this app pins in
# .streamlit/config.toml -- all-pairs colour-blind separation dE 9.2, normal-vision dE
# 24.0, both clear of their floors. The aqua slot measures 2.74:1 against that surface,
# under the 3:1 bar, so every chart using it carries direct labels and a table view; see
# `outcome_chart`. Do not substitute a colour here without re-running that check: the
# numbers above are true of these three hexes on this surface, and of nothing else.
SERIES_RECOVERED = "#2a78d6"   # slot 1, blue
SERIES_ESCALATED = "#eb6834"   # slot 2, orange
SERIES_ABANDONED = "#1baf7a"   # slot 3, aqua
CHART_SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

MIN_LABELLED_SHARE = 0.07
"""How large a stacked segment must be, as a share of its bar, to carry its number inside
it.

A count drawn inside a segment narrower than its own digits gets clipped, which looks like
a rendering bug and is unreadable either way. Smaller segments therefore carry no inside
label: the value stays reachable on hover and in the comparison table. This is also what
keeps the chart from putting a number on every single mark."""

STATUS_LABELS = {
    TransactionStatus.RECOVERED: "recovered",
    TransactionStatus.ESCALATED_TO_HUMAN: "escalated to a human",
    TransactionStatus.ABANDONED: "abandoned",
    TransactionStatus.IN_PROGRESS: "still in progress",
}

FEED_TAIL_LENGTH = 12
"""How many of the most recent sentences the live feed shows at once.

A demo batch produces a few thousand entries, and re-rendering all of them on every
update would make the feed quadratic in the batch size and unreadable besides. Only the
tail is *displayed live*; the whole trail is kept in full and stays available in the
per-transaction drill-down below, so nothing is discarded — only the on-screen window is
bounded.
"""


FEED_SEPARATOR = "\n\n---\n\n"
"""What sits between two entries in the live feed: a blank line, a horizontal rule,
a blank line. Each sentence is a dense paragraph of reasoning, so without a visible
divider a fast-moving feed reads as one run-on wall of text."""


# ---------------------------------------------------------------------------
# Run configuration and state
# ---------------------------------------------------------------------------


@dataclass
class RunSettings:
    """Everything the sidebar controls, in one object.

    `simulator` is the world; `narration_delay_seconds` is the only field that is not
    part of the run at all — see `run_live` for why it exists and why it cannot change a
    single outcome.
    """

    simulator: SimulatorConfig
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    narration_delay_seconds: float = 0.0


@dataclass
class CompletedRun:
    """What one finished batch leaves behind for the rest of the page to read.

    Held in `st.session_state` so that Streamlit's re-run-the-whole-script-on-every-
    interaction model does not re-run the *batch* every time somebody picks a different
    transaction from a dropdown. The run happens once, when the button is pressed; every
    interaction after that reads this object.
    """

    paired: PairedRun
    trail: AuditTrail
    settings: RunSettings
    """The sidebar settings this run was launched with.

    Kept so the export can name its files after the seed a person actually typed, without
    reaching back into the widgets — which by then may have been nudged to something else
    while the results on screen still belong to the old run."""

    key: str
    """`metrics.run_key` for this run: a fingerprint of the simulator, orchestrator and
    baseline configurations together.

    Computed once here rather than read off `results`, because a run that could not be
    scored still has a provenance — and the export needs to name its files even then.
    `BatchResults.key` is this same function over these same inputs, so the two always
    agree; `tests/test_dashboard.py` pins that rather than leaving it as a claim."""

    results: Optional[BatchResults]
    """The Metrics engine's (C3) scoring of this exact run, computed once when the batch
    finishes — or `None` when there was nothing to score.

    `None` is reachable from the sidebar: a small population at a low failure rate can
    produce a world in which no billing event failed at all. That is a perfectly valid
    run, and it is not a batch with a recovery rate of zero — there is no denominator, so
    there is no rate. The Metrics engine refuses to summarise an empty batch for exactly
    that reason, and this field carries the same refusal rather than inventing a score.

    Held here rather than recomputed per view so that every number on the page comes from
    one object: the head-to-head tiles, the against-truth view and the download button are
    then guaranteed to be describing the same run rather than three separately-derived
    copies of it. It is also what carries the run's provenance (`key`, `label`) into
    anything this page displays or exports."""


# ---------------------------------------------------------------------------
# The run controls (C4, operation 5)
# ---------------------------------------------------------------------------
#
# "Provide a way to trigger a fresh batch run with a chosen seed/configuration, for live
# demo control."
#
# Task 1 built the minimum the feed needed — seed, population, failure rate, a Run button.
# This is the rest of it, and the item that mattered most was the **mass-failure
# scenario**: `SimulatorConfig` has carried that mode since Phase 2 (A2 operation 10) and
# nothing in the UI had ever switched it on.
#
# **Switching it on is how a defect got found.** A2 operation 10 describes the mode as
# forcing a share of billing events "onto the exact same simulated instant" — a thundering
# herd for Jitter (IDEA.md §8c) to spread out. It did force them onto one instant. What
# nobody had noticed is that they were *already* there: every customer was generated with
# the same billing cycle length and the same start date, so every charge in a cycle fell on
# one instant, toggle or no toggle, and the spaced-out "normal batch" the mode is
# documented as contrasting against did not exist. The toggle was a volume dial wearing a
# synchronisation label.
#
# Fixed in Phase 13: customers now renew on their own anniversary within the cycle
# (`Customer.billing_anniversary_offset_days`), so the toggle changes both the volume and
# the timing, and the help text below can describe it without a caveat. A 100-customer
# world at a 25% base rate goes from 29 charging instants with 7 on the busiest, to a
# forced 31 on one mid-cycle instant. `tests/test_dashboard.py` now asserts both halves.
#
# Everything here is *input*. This section chooses what world to build and how the agents
# are configured; it computes nothing about the outcome, and it cannot: the run key shown
# below is a fingerprint of the configuration alone.


PRESET_KEY = "preset"

STANDARD_PRESET = "Standard world"
MASS_FAILURE_PRESET = "Mass failure"
QUIET_PRESET = "Quiet world"
CUSTOM_PRESET = "Custom"

PRESETS: dict[str, dict[str, Any]] = {
    STANDARD_PRESET: {
        "seed": 11,
        "customers": 200,
        "failure_rate": 0.25,
        "mass_failure": False,
        "mass_fraction": 0.30,
    },
    MASS_FAILURE_PRESET: {
        "seed": 11,
        "customers": 200,
        "failure_rate": 0.25,
        "mass_failure": True,
        "mass_fraction": 0.30,
    },
    QUIET_PRESET: {
        "seed": 3,
        "customers": 20,
        "failure_rate": 0.05,
        "mass_failure": False,
        "mass_fraction": 0.30,
    },
}
"""One click each for the three worlds worth showing, so a live demo is not four sliders
set while people watch.

* **Standard world** — the configuration every number recorded in `notes/TRACKER.md` was
  produced under. This is the one to quote from.
* **Mass failure** — the same world with the mass-failure scenario on: a third of the
  population's charges are dragged onto one mid-cycle instant and forced to fail
  regardless of the base rate, roughly doubling how many failures the pacing has to spread
  out at once. This is a genuine retry storm against an otherwise spread-out batch. One
  caveat still worth stating if a judge presses: the pile-up is real in the data, but the
  batch is processed one transaction at a time, so the agent does not *experience* it as
  simultaneous pressure — see notes/TRACKER.md's "system-wide is processing-order" note.
* **Quiet world** — a small population at a low failure rate, where nothing fails at all.
  Worth having as a preset rather than hiding: it is the run that has no recovery rate,
  and watching the page say "there is no denominator" instead of "0%" is the honesty
  claim in this project being demonstrated rather than asserted.

Deliberately *not* in any preset: the narration pause. It changes the screen, never the
run, so switching worlds should not reset it.
"""

AGENT_DEFAULTS = {
    "epsilon": OrchestratorConfig().epsilon,
    "horizon": OrchestratorConfig().recovery_horizon_days,
    "agent_seed": OrchestratorConfig().seed,
    "baseline_interval": BaselineConfig().retry_interval_hours,
    "baseline_attempts": BaselineConfig().max_attempts,
}
"""Read off the dataclasses rather than restated, so this panel cannot drift away from
the defaults every recorded number was produced under."""


def _apply_preset() -> None:
    """Copy the chosen preset's values into the widget state.

    Runs as the preset selector's `on_change` callback, which is the only point at which
    Streamlit allows a widget's value to be set from code. `Custom` deliberately does
    nothing: it means "leave what is there", which is what somebody selecting it after
    moving a slider wants.
    """
    values = PRESETS.get(st.session_state.get(PRESET_KEY))
    if values is not None:
        st.session_state.update(values)


def _initialise_run_state() -> None:
    """Seed the widget state from the standard preset on the first render only."""
    defaults = {
        PRESET_KEY: STANDARD_PRESET,
        "narration_delay": 0.10,
        **PRESETS[STANDARD_PRESET],
        **AGENT_DEFAULTS,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _matches_preset(name: str) -> bool:
    """Whether the current widget state still is the preset it claims to be."""
    values = PRESETS.get(name)
    if values is None:
        return True
    return all(st.session_state.get(key) == value for key, value in values.items())


def sidebar_settings() -> RunSettings:
    """The run controls (ARCHITECTURE.md C4, operation 5).

    Returns the configuration a Run press would use. Reading this function is also the
    quickest way to see what a run *is*: a `SimulatorConfig` (the world), an
    `OrchestratorConfig` (the Smart Agent), a `BaselineConfig` (the control group), and one
    display-only number that is part of none of them.
    """
    _initialise_run_state()
    st.sidebar.header("Run a batch")

    st.sidebar.selectbox(
        "Preset",
        [STANDARD_PRESET, MASS_FAILURE_PRESET, QUIET_PRESET, CUSTOM_PRESET],
        key=PRESET_KEY,
        on_change=_apply_preset,
        help="Three worlds worth showing, plus Custom for anything else. Picking one "
        "sets the controls below; moving a control afterwards is fine and is reported "
        "as modified.",
    )

    st.sidebar.number_input(
        "Seed",
        min_value=0,
        max_value=999_999,
        step=1,
        key="seed",
        help=(
            "The world. The same seed always produces the same customers, the same "
            "billing events and the same hidden truths — so both agents can be compared "
            "on identical ground, and any run here can be reproduced later."
        ),
    )
    st.sidebar.slider(
        "Customers",
        min_value=20,
        max_value=1000,
        step=20,
        key="customers",
        help="How large a population to generate. Only the failed billing events are retried.",
    )
    st.sidebar.slider(
        "Base failure rate",
        min_value=0.05,
        max_value=0.50,
        step=0.05,
        key="failure_rate",
        help=(
            "How often a billing event fails in the first place. ~8% is the commonly "
            "cited industry ballpark; the demo default is raised to 25% so a modest "
            "population still produces enough failures to compare."
        ),
    )

    st.sidebar.toggle(
        "Mass failure (bank outage)",
        key="mass_failure",
        help=(
            "Normally each customer renews on their own anniversary, so a batch's "
            "charges are spread across the whole billing cycle. This forces a share of "
            "them onto one mid-cycle instant *and* fails them regardless of the base "
            "failure rate - a bank outage on renewal day - which is the thundering-herd "
            "condition the jitter exists to spread back out."
        ),
    )
    if st.session_state["mass_failure"]:
        st.sidebar.slider(
            "Share failing at once",
            min_value=0.10,
            max_value=1.00,
            step=0.10,
            key="mass_fraction",
            help="How much of the population is caught in the outage.",
        )

    with st.sidebar.expander("Agent configuration"):
        st.caption(
            "Left alone, these are the values every number in `notes/TRACKER.md` was "
            "recorded under. Changing one is a different experiment - not a worse one, "
            "but the recorded figures no longer describe it, and the run key below "
            "changes to say so."
        )
        st.number_input(
            "Smart Agent decision seed",
            min_value=0,
            max_value=999_999,
            step=1,
            key="agent_seed",
            help="Seeds the agent's *own* choices - arm selection and jitter - separately "
            "from the world's seed, so 'same seed, same decisions' holds too.",
        )
        st.slider(
            "Explore rate (epsilon)",
            min_value=0.0,
            max_value=0.5,
            step=0.05,
            key="epsilon",
            help="How often the bandit tries an arm other than its current best. At 0 it "
            "stops learning and rides whatever looked good first.",
        )
        st.slider(
            "Recovery horizon (days)",
            min_value=1.0,
            max_value=60.0,
            step=1.0,
            key="horizon",
            help="How long after the failed charge either agent keeps trying before "
            "giving up as out of time. Both agents share it, so 'abandoned' means the "
            "same thing on both sides of the comparison.",
        )
        st.divider()
        st.caption("The Baseline Agent's whole policy is these two numbers.")
        st.slider(
            "Baseline retry interval (hours)",
            min_value=1.0,
            max_value=48.0,
            step=1.0,
            key="baseline_interval",
            help="The flat wait between the Baseline's retries - no adaptation, the same "
            "gap every time.",
        )
        st.slider(
            "Baseline attempt cap",
            min_value=1,
            max_value=25,
            step=1,
            key="baseline_attempts",
            help="How many times the Baseline retries before handing over to a human.",
        )

    st.sidebar.divider()
    st.sidebar.slider(
        "Narration pause (seconds per transaction)",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="narration_delay",
        help=(
            "Display speed only. Simulated time is instant, so the whole batch finishes "
            "in about a second — too fast to read. This pauses the *screen* between "
            "transactions. It changes nothing about the run: same seed, same decisions, "
            "same results, with or without it."
        ),
    )

    settings = RunSettings(
        simulator=SimulatorConfig(
            seed=int(st.session_state["seed"]),
            num_customers=int(st.session_state["customers"]),
            base_failure_rate=float(st.session_state["failure_rate"]),
            mass_failure_scenario=bool(st.session_state["mass_failure"]),
            # Normalised to the default when the scenario is off. The fraction changes
            # nothing in that case, and letting an inert field into the fingerprint would
            # hand two byte-identical runs two different keys — which is the one thing a
            # key must never do, since it is what tells a reader "these numbers came from
            # the same place".
            mass_failure_fraction=(
                float(st.session_state["mass_fraction"])
                if st.session_state["mass_failure"]
                else SimulatorConfig(seed=0).mass_failure_fraction
            ),
        ),
        orchestrator=OrchestratorConfig(
            seed=int(st.session_state["agent_seed"]),
            epsilon=float(st.session_state["epsilon"]),
            recovery_horizon_days=float(st.session_state["horizon"]),
        ),
        baseline=BaselineConfig(
            retry_interval_hours=float(st.session_state["baseline_interval"]),
            max_attempts=int(st.session_state["baseline_attempts"]),
            recovery_horizon_days=float(st.session_state["horizon"]),
        ),
        narration_delay_seconds=float(st.session_state["narration_delay"]),
    )

    render_run_provenance(settings)
    return settings


def render_run_provenance(settings: RunSettings) -> None:
    """What the configuration in the sidebar *is*, stated before it is run.

    The run key can be computed from the inputs alone, so there is no reason to make
    somebody run a batch to find out which run they are about to look at. Two things are
    said here that a caption elsewhere could not say as honestly:

    - **Whether this is still the standard configuration.** Every figure in
      `notes/TRACKER.md` and in this project's notes was recorded under one specific
      configuration. A judge who nudges the explore rate and then reads a recorded number
      off a slide is comparing two different experiments. The page says so at the moment
      the nudge happens, rather than leaving the mismatch to be discovered later or never.
    - **That the narration pause is not part of it.** It is the one control here that
      cannot change a result, and saying so is what makes the rest of the claim credible.
    """
    key = metrics.run_key(settings.simulator, settings.orchestrator, settings.baseline)
    selected = st.session_state.get(PRESET_KEY, CUSTOM_PRESET)
    modified = not _matches_preset(selected)

    agents_changed = [
        name
        for name, default in AGENT_DEFAULTS.items()
        if st.session_state.get(name) != default
    ]

    st.sidebar.caption(f"This configuration is run `{key}`.")
    if modified:
        st.sidebar.caption(f"Modified from **{selected}**.")
    if agents_changed:
        st.sidebar.warning(
            "The agents are no longer at their recorded settings, so the figures in "
            "`notes/NOTES.md` and `notes/TRACKER.md` describe a different experiment "
            "from this one. The run key above has changed to say so.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# The live feed (C4, operation 1)
# ---------------------------------------------------------------------------


def format_entry(entry: AuditEntry) -> str:
    """One trail entry as a line of the feed.

    Everything here except the punctuation came off the entry: the simulated timestamp,
    the agent label the trail attached at collection time, the transaction id, and the
    sentence the renderer produced. No summarising, no rewording — a judge reading this
    line on screen and the same line in the exported CSV must see the same claim.
    """
    stamp = entry.occurred_at.strftime("%Y-%m-%d %H:%M") if entry.occurred_at else "--"
    agent = AGENT_LABELS.get(entry.agent, entry.agent)
    return f"**`{stamp}`** · *{agent}* · `{entry.transaction_id}`  \n{entry.sentence}"


@dataclass
class FeedTally:
    """The running counters beside the feed, accumulated one finished transaction at a
    time.

    Deliberately counted from each result's own recorded `status` and attempt list rather
    than re-derived from anything: these are the same facts the Metrics engine (C3) will
    aggregate properly once the batch completes, so the two must not be able to disagree.
    They are shown live because watching them move is the point of the feed — not because
    this is where the headline numbers come from.
    """

    transactions: int = 0
    attempts: int = 0
    recovered: int = 0
    escalated: int = 0
    abandoned: int = 0

    def add(self, result: AgentResult) -> None:
        self.transactions += 1
        self.attempts += len(result.transaction.attempts)
        status = result.transaction.status
        if status == TransactionStatus.RECOVERED:
            self.recovered += 1
        elif status == TransactionStatus.ESCALATED_TO_HUMAN:
            self.escalated += 1
        elif status == TransactionStatus.ABANDONED:
            self.abandoned += 1


def render_tally(placeholder: "st.delta_generator.DeltaGenerator", tally: FeedTally) -> None:
    """The live counters, redrawn into a placeholder each time one changes.

    Written through `st.empty()` rather than appended to a container, because Streamlit
    containers *accumulate* — drawing five metrics into the same container once per
    transaction would leave a page several hundred rows long by the end of a batch.

    Counts, not rates. A recovery *rate* quoted off a batch that is still running is not
    the run's recovery rate, and the head-to-head view (C4 operation 2) is where that
    number properly belongs.
    """
    with placeholder.container():
        columns = st.columns(5)
        columns[0].metric("Transactions", tally.transactions)
        columns[1].metric("Attempts", tally.attempts)
        columns[2].metric("Recovered", tally.recovered)
        columns[3].metric("Escalated", tally.escalated)
        columns[4].metric("Abandoned", tally.abandoned)


def run_live(settings: RunSettings) -> CompletedRun:
    """Run both agents over one seeded world, narrating each transaction as it finishes
    (ARCHITECTURE.md C4, operation 1).

    The batch is driven by `baseline_agent.iter_paired_batch`, which yields each finished
    transaction and returns the assembled `PairedRun` at the end. That is a single pass:
    the sentences on screen and the metrics computed afterwards describe the same run,
    not two runs that happen to share a seed.

    **What "live" does and does not mean here.** Each transaction is narrated the moment
    the agent finishes it, which is genuinely as-it-happens. But the batch is processed
    one transaction to completion at a time (D1's loop, unchanged), so the feed reads in
    *production* order — one customer's whole journey, then the next customer's — not in
    simulated-clock order. That is the right order for narrating a decision chain, and it
    is why the drill-down below reads as a story. It also means the feed is not a picture
    of what the whole population was doing hour by hour; `AuditTrail.by_simulated_time`
    is that view, and `notes/TRACKER.md` records the same caveat against the Pacing
    component's "system-wide" claim.

    This loop cannot influence the run. It observes results that are already final and
    accumulates them; the generator's own bookkeeping is what builds the `PairedRun`.
    """
    trail = AuditTrail()
    tally = FeedTally()

    status_box = st.empty()
    status_box.info("Building the world…")
    tally_box = st.empty()
    st.caption(
        f"Live decision feed — the {FEED_TAIL_LENGTH} most recent entries. "
        "The full log is kept and is browsable per transaction once the run finishes."
    )
    feed_box = st.empty()

    render_tally(tally_box, tally)

    stream = iter_paired_batch(
        settings.simulator,
        orchestrator_config=settings.orchestrator,
        baseline_config=settings.baseline,
    )

    lines: list[str] = []
    paired: Optional[PairedRun] = None

    while True:
        try:
            result = next(stream)
        except StopIteration as finished:
            paired = finished.value
            break

        entries = trail.collect_transaction(result)
        tally.add(result)

        # Newest block on top so the feed never needs scrolling to follow, chronological
        # within each transaction so its reasoning still reads in the order it happened.
        lines = ([format_entry(entry) for entry in entries] + lines)[:FEED_TAIL_LENGTH]
        feed_box.markdown(FEED_SEPARATOR.join(lines))
        render_tally(tally_box, tally)
        status_box.info(
            f"Running the {AGENT_LABELS[agent_for(result)]} — "
            f"{tally.transactions} transactions narrated so far."
        )

        if settings.narration_delay_seconds:
            time.sleep(settings.narration_delay_seconds)

    # No progress *bar*: both agents' totals are only known once the world has been built,
    # which happens inside the generator, and a bar advancing on a denominator this file
    # guessed would be a made-up number on a page whose whole argument is that it does not
    # make numbers up.
    status_box.success(
        f"Finished — {tally.transactions} transactions across both agents, "
        f"{tally.attempts} attempts."
    )
    assert paired is not None  # the generator always returns one; narrows the type
    # Scored once, here, from the run that was just narrated -- not from a fresh batch.
    # A world in which nothing failed has nothing to score; see `CompletedRun.results`.
    results = metrics.evaluate(paired) if paired.smart else None
    return CompletedRun(
        paired=paired,
        trail=trail,
        settings=settings,
        key=metrics.run_key(
            paired.simulator_config,
            paired.orchestrator_config,
            paired.baseline_config,
        ),
        results=results,
    )


# ---------------------------------------------------------------------------
# The head-to-head comparison (C4, operation 2)
# ---------------------------------------------------------------------------


METRIC_LABELS = {
    "transactions": "Failed payments handled",
    "recovered": "Recovered",
    "escalated_to_human": "Escalated to a human",
    "abandoned": "Abandoned (ran out of time)",
    "unresolved": "Still unresolved (should be 0)",
    "recovery_rate": "Recovery rate",
    "escalation_rate": "Escalation rate",
    "abandonment_rate": "Abandonment rate",
    "total_attempts": "Total retry attempts",
    "mean_attempts_all": "Mean attempts per payment",
    "mean_attempts_to_recovery": "Mean attempts per recovery",
    "median_attempts_to_recovery": "Median attempts per recovery",
    "circuit_breaker_consulted": "Consulted the circuit breaker",
    "channel_closures": "Channels permanently retired",
    "transactions_with_a_closure": "Payments that lost a channel",
    "closures_by_channel": "Closures by channel",
    "terminal_reasons": "How journeys ended",
}
"""Readable names for the rows of the comparison table.

Renaming is the *only* thing this map is allowed to do. A metric with no entry here keeps
its raw name and is still shown: dropping a row because nobody wrote a label for it would
let a future metric go missing from the table without anybody noticing.
"""

RATE_METRICS = frozenset(
    {"recovery_rate", "escalation_rate", "abandonment_rate"}
)
"""Which rows are proportions, and so are shown as percentages rather than as `0.6835`."""


def format_cell(metric: str, value: Any) -> str:
    """One comparison-table cell, formatted for reading.

    Non-numeric cells pass through untouched — which is the point of this function
    existing rather than a blanket number format. `HeadToHead.to_frame()` has already put
    the words "not applicable (no circuit breaker)" in the Baseline's closure cells, and
    that text must survive to the screen exactly as written: it is the one cell in this
    whole table that a reader could most easily misread, and the metrics engine says so
    in words precisely so the table does not depend on a caption nobody reads.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if metric in RATE_METRICS:
            return f"{value:.1%}"
        if isinstance(value, float):
            return f"{value:.2f}"
        return f"{value:,}"
    return str(value)


def stacked_comparison_chart(
    row_labels: Sequence[str],
    series: Sequence[tuple[str, str, Sequence[int]]],
    hover_noun: str = "payments",
) -> "go.Figure":
    """A horizontal stacked bar comparing the same part-to-whole breakdown across agents.

    The one chart shape this page uses, built once and called twice, so that every
    comparison on the page is drawn to the same rules rather than to whichever rules the
    author remembered on the day.

    Those rules, and why each is here:

    - **Part-to-whole across two groups is what a stacked bar is for.** Both charts on
      this page answer "of the payments handed to each agent, what share ended each way",
      and both are honest by geometry: the bars are the same length by construction,
      because both agents were handed the identical set of payments.
    - **Direct labels, but only where one fits** (`MIN_LABELLED_SHARE`). A number drawn
      inside a segment narrower than its own digits is clipped; the value stays on hover
      and in the table below. This is also the relief the low-contrast palette slot needs
      — nothing here rests on telling two fills apart.
    - **Recessive axes.** The marks carry their own numbers, so the scale underneath is
      scaffolding and does not compete with the data.

    `series` is `(name, colour, counts)` with one count per entry in `row_labels`, in the
    same order. A `None` count is not accepted anywhere on this page: see the callers.
    """
    totals = [
        sum(counts[index] for _, _, counts in series)
        for index in range(len(row_labels))
    ]

    figure = go.Figure()
    for name, colour, counts in series:
        figure.add_bar(
            y=list(row_labels),
            x=list(counts),
            name=name,
            orientation="h",
            marker_color=colour,
            # A 2px surface-coloured gap between adjacent fills, so two segments never
            # merge into one block at a glance.
            marker_line_color=CHART_SURFACE,
            marker_line_width=2,
            text=[
                f"{count:,}"
                if totals[index] and count / totals[index] >= MIN_LABELLED_SHARE
                else ""
                for index, count in enumerate(counts)
            ],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=13),
            cliponaxis=False,
            hovertemplate="%{y}<br>" + name + ": %{x:,} " + hover_noun + "<extra></extra>",
        )

    figure.update_layout(
        barmode="stack",
        # Tall enough to hold the plot *and* the legend band above it, rather than sizing
        # to the plot and letting the legend crop.
        height=230,
        margin=dict(l=0, r=0, t=44, b=0),
        # A legend is always present for two or more series.
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None
        ),
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font=dict(color=TEXT_PRIMARY),
        # Thin marks: two full-width saturated blocks would read loud, so the bars stay
        # narrow with room around them.
        bargap=0.55,
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(
        showgrid=False,
        zeroline=False,
        ticksuffix="  ",
        tickfont=dict(color=TEXT_SECONDARY),
    )
    return figure


def outcome_chart(head: "metrics.HeadToHead") -> "go.Figure":
    """What happened to the same set of failed payments under each agent.

    **Why this chart and not a grouped bar of every metric.** The question a viewer is
    actually asking here is part-to-whole: of the payments each agent was handed, what
    share ended each way? A stacked bar answers that in one glance and keeps both agents
    on one shared scale — which is honest precisely because both agents were handed the
    *same* payments, so the two bars are the same length by construction. The headline
    numbers above it are a row of stat tiles rather than more bars, because a handful of
    single values is not a chart.

    **What is deliberately not on it.** Channel closures. The Baseline Agent has no
    circuit breaker, so its closure count is *absent*, not zero — and a bar chart has no
    way to draw "absent" that a reader will not read as zero. That number appears only in
    the table below, where it can state its own reason in words. This is the single
    easiest way for this page to tell a lie, and the rule is simply: a metric only one
    agent has never goes on a chart that shows both.

    Colours are the first three slots of the project's categorical palette, in fixed
    order, validated with the palette checker against this app's pinned light surface
    (`#fcfcfb`): all-pairs colour-blind separation dE 9.2, normal-vision dE 24.0. The aqua
    slot sits below 3:1 contrast against that surface, so the "relief rule" applies — the
    counts are drawn directly on the segments and the full table is one click away.
    """
    agents = [
        (AGENT_LABELS[AGENT_BASELINE], head.baseline),
        (AGENT_LABELS[AGENT_SMART], head.smart),
    ]
    outcomes = [
        ("Recovered", SERIES_RECOVERED, "recovered"),
        ("Escalated to a human", SERIES_ESCALATED, "escalated_to_human"),
        ("Abandoned", SERIES_ABANDONED, "abandoned"),
    ]
    return stacked_comparison_chart(
        [label for label, _ in agents],
        [
            (name, colour, [getattr(summary, field_name) for _, summary in agents])
            for name, colour, field_name in outcomes
        ],
    )


def render_head_to_head(run: CompletedRun) -> None:
    """The headline Baseline-vs-Smart comparison (ARCHITECTURE.md C4, operation 2).

    Reads the finished `BatchResults` the run already produced. Nothing is recomputed
    here and nothing is re-derived: every figure on screen is a field on an object the
    Metrics engine (C3) built, so the page and the exported JSON cannot disagree.
    """
    st.subheader("Head to head")
    if run.results is None:
        st.info(
            "No billing event failed in this world, so there is nothing to recover and "
            "nothing to compare. That is not a recovery rate of zero - there is no "
            "denominator to compute one from. Try a larger population or a higher "
            "failure rate."
        )
        return

    head = run.results.head_to_head
    smart, baseline = head.smart, head.baseline

    st.caption(
        f"Both agents over one identical world - {run.results.label}. "
        "Same customers, same failed payments, same hidden truths."
    )

    columns = st.columns(4)
    columns[0].metric(
        "Recovery rate",
        f"{smart.recovery_rate:.1%}",
        delta=f"{head.recovery_rate_gain_points:+.1f} pts vs baseline",
        help="Share of failed payments the Smart Agent got paid. The Baseline's own rate "
        f"over the same payments was {baseline.recovery_rate:.1%}.",
    )
    columns[1].metric(
        "Retry attempts used",
        f"{smart.total_attempts:,}",
        delta=f"{-head.attempt_reduction:.1%} vs baseline",
        delta_color="inverse",
        help="Every retry submitted across the batch. Fewer is better: each one is a "
        "request on the payment network and a step towards getting the card blocked. "
        f"The Baseline used {baseline.total_attempts:,}.",
    )
    columns[2].metric(
        "Escalated to a human",
        f"{smart.escalated_to_human:,}",
        delta=f"{smart.escalated_to_human - baseline.escalated_to_human:+,} vs baseline",
        delta_color="inverse",
        help="Payments handed to a person because the agent ran out of safe options. "
        f"The Baseline escalated {baseline.escalated_to_human:,}.",
    )

    # The one tile that must not have a delta. The Baseline has no circuit breaker, so
    # there is no number on its side to subtract — "43 vs 0" would be read as two agents
    # that both counted, and only one of them did. The cell says so in words instead.
    closures = smart.channel_closures
    columns[3].metric(
        "Channels permanently retired",
        f"{closures:,}" if closures is not None else "-",
        help="Payment methods the circuit breaker walled off for good, to stop the agent "
        "retrying a card that is never going to work.",
    )
    columns[3].caption(f"Baseline: {metrics.NOT_APPLICABLE_NO_BREAKER}")

    st.plotly_chart(outcome_chart(head), use_container_width=True, theme=None)

    with st.expander("Every metric, side by side"):
        frame = head.to_frame()
        table = pd.DataFrame(
            {
                "Metric": [METRIC_LABELS.get(m, m) for m in frame["metric"]],
                AGENT_LABELS[AGENT_SMART]: [
                    format_cell(m, v) for m, v in zip(frame["metric"], frame[AGENT_SMART])
                ],
                AGENT_LABELS[AGENT_BASELINE]: [
                    format_cell(m, v)
                    for m, v in zip(frame["metric"], frame[AGENT_BASELINE])
                ],
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Rendered from the Metrics engine's own comparison table. Cells reading "
            f'"{metrics.NOT_APPLICABLE_NO_BREAKER}" are not zeros - the Baseline Agent '
            "never consulted that mechanism, so it has no number to report there."
        )


# ---------------------------------------------------------------------------
# Against the hidden truth (C4, operation 3)
# ---------------------------------------------------------------------------
#
# The secondary, more detailed view. The head-to-head above says which agent did better;
# this says how each did against what was *actually possible* — scored on the answer key
# the Simulator has held privately since the world was generated, and that no decision-
# making component has ever been allowed to see.
#
# **This is the most over-claimable view on the page**, and the reason belongs next to the
# code rather than in a notes file. These numbers read as "the agent was right", so three
# of them need reading instructions attached, permanently:
#
# 1. `label_agreement` is **1.0 by construction**. The Simulator picks a decline code
#    *because* it has already decided the failure is soft or hard, drawing on the same
#    JSON registries the Classifier later reads; the two could not disagree unless one
#    were broken. It is a regression canary, not a score — so it is deliberately *not*
#    rendered as a stat tile, because a tile reading "100%" is what gets screenshotted.
# 2. The hard call asserts a **channel** is dead, not that a transaction is unrecoverable.
#    A hard-declined customer recovered by switching to their UPI mandate is not a
#    classifier error, and scoring it as one would be a false accusation of the one
#    component that got it right. So the recoverability figures are labelled as the
#    conditional base rates they are.
# 3. `wasted_attempt_share` reads backwards — the Smart Agent's share is *higher* than the
#    Baseline's while its absolute waste is lower, because it cut its total attempts
#    faster than it cut its wasted ones. It is quoted only next to the absolute numbers.
#
# And the rule task 2 established applies again here, in a new place: `AgainstTruth`
# reports `classifier=None` for the Baseline, exactly as the closure metrics are `None`.
# Operation 7 therefore gets its own clearly-labelled Smart-Agent-only subsection, with no
# Baseline column, no delta and no chart.


NOT_RECORDED_BY_THIS_AGENT = "not a reason this agent can record"
"""What a missed-reason cell says when the tag belongs to the other agent's vocabulary.

The two agents stop for different reasons because they have different mechanisms: the
Baseline's journeys can only ever end at its fixed attempt cap, while the Smart Agent's
can end at a circuit-breaker closure or at the recovery horizon. A `0` in the other
agent's column would read as "this agent avoided that ending" rather than "that ending
does not exist for this agent" — the same failure mode as a closure count of zero.
"""

TRUTH_METRIC_LABELS = {
    "transactions": "Failed payments handled",
    "scored": "Payments with an answer key",
    "recoverable": "Genuinely recoverable",
    "unrecoverable": "Genuinely unrecoverable",
    "recovered_of_recoverable": "Recovered, of the recoverable",
    "recall": "Recall (share of the possible, captured)",
    "missed_recoverable": "Recoverable, but missed",
    "recovered_of_unrecoverable": "Recovered the impossible (must be 0)",
    "wasted_attempts": "Attempts spent on lost causes",
    "mean_wasted_attempts": "Mean attempts per lost cause",
    "max_wasted_attempts": "Most attempts on a single lost cause",
    "wasted_attempt_share": "Lost-cause attempts, as a share of all attempts",
    "calls": "Opening diagnoses recorded",
    "soft_calls": "Called soft (temporary)",
    "hard_calls": "Called hard (this channel is dead)",
    "unrecognized_calls": "Refused to classify (code not in the registry)",
    "label_agreement": "Agreement with the simulator's own label [canary]",
    "label_disagreements": "Disagreements with the simulator's label [canary]",
    "recoverable_given_soft": "Base rate: recoverable, given a soft call",
    "recoverable_given_hard": "Base rate: recoverable elsewhere, given a hard call",
    "hard_calls_on_a_dead_channel": "Hard calls the answer key agrees were dead channels",
    "hard_call_channel_precision": "Hard-call channel precision [canary]",
    "attempts_on_a_dead_origin_channel": "Retries onto a channel it had called dead",
}
"""Readable names for the diagnostic table. Rename only — the same rule as
`METRIC_LABELS`: a metric with no entry here keeps its raw name and is still shown,
because a row dropped for want of a label goes missing without anybody noticing.

`[canary]` is carried in the label itself rather than in a caption underneath, because a
caption is what gets cropped out of a screenshot while the row it explains survives.
"""

TRUTH_RATE_METRICS = frozenset(
    {
        "recall",
        "wasted_attempt_share",
        "label_agreement",
        "recoverable_given_soft",
        "recoverable_given_hard",
        "hard_call_channel_precision",
    }
)

SERIES_MISSED = SERIES_ESCALATED
"""Slot 2 of the same categorical palette, reused for "missed" on the recall chart.

Safe without re-validation, and only because it is a *subset* of an already-validated
set: if all three slots clear the separation floors pairwise, any two of them do. Slot 1
keeps meaning "recovered" on both charts, which is the consistency that matters. A fourth
colour anywhere on this page would need the full check re-run — see the palette note at
the top of this file."""


def _delta_points(smart: Optional[float], baseline: Optional[float]) -> Optional[str]:
    """A percentage-point delta, or nothing at all if either side is absent.

    Absent means absent: there is no arithmetic to perform against a number that was
    never computed, and a delta rendered from one is a comparison the page invented.
    """
    if smart is None or baseline is None:
        return None
    return f"{(smart - baseline) * 100:+.1f} pts vs baseline"


def format_truth_cell(metric: str, value: Any) -> str:
    """One diagnostic-table cell. Words pass through untouched, as in `format_cell`."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if metric in TRUTH_RATE_METRICS:
            return f"{value:.1%}"
        if isinstance(value, float):
            return f"{value:.2f}"
        return f"{value:,}"
    return str(value)


def recall_chart(
    smart: "metrics.AgainstTruth", baseline: "metrics.AgainstTruth"
) -> "go.Figure":
    """Of the payments that genuinely could have been recovered, how many each agent got.

    The same part-to-whole shape as the outcome chart, and honest by the same geometry:
    the two bars are equal by construction, because which payments were recoverable is a
    property of the *world*, fixed before either agent started. Both series exist for both
    agents, so nothing here breaks the rule that a single-agent metric never goes on a
    shared chart — which is exactly why the Classifier's numbers are further down, in
    words, and on no chart at all.
    """
    agents = [
        (AGENT_LABELS[AGENT_BASELINE], baseline),
        (AGENT_LABELS[AGENT_SMART], smart),
    ]
    return stacked_comparison_chart(
        [label for label, _ in agents],
        [
            (
                "Recovered",
                SERIES_RECOVERED,
                [scored.recovered_of_recoverable for _, scored in agents],
            ),
            (
                "Missed",
                SERIES_MISSED,
                [scored.missed_recoverable for _, scored in agents],
            ),
        ],
        hover_noun="recoverable payments",
    )


def _missed_reasons_frame(scored: "metrics.AgainstTruth") -> pd.DataFrame:
    """Why one agent's recoverable payments got away, in that agent's own reason tags,
    rendered through the Audit Trail's English so the table and the narration agree."""
    counts = scored.missed_by_terminal_reason
    return pd.DataFrame(
        {
            "Why it got away": [reason_phrase(tag) for tag in counts],
            "Payments": [counts[tag] for tag in counts],
        }
    )


def render_against_truth(run: CompletedRun) -> None:
    """The against-hidden-truth diagnostic view (ARCHITECTURE.md C4, operation 3).

    Reads `run.results` and computes nothing. Every figure is a field on the
    `AgainstTruth` records the Metrics engine (C3) already built for this exact run, so
    this view, the head-to-head above it and the export cannot disagree.
    """
    st.subheader("Against the answer key")
    if run.results is None:
        st.info(
            "Nothing failed in this world, so there is nothing to score against the "
            "hidden truth."
        )
        return

    smart = run.results.smart_vs_truth
    baseline = run.results.baseline_vs_truth

    st.caption(
        "Before either agent started, the simulator decided which of these payments were "
        "recoverable at all, on which channel, and after how long. Neither agent could "
        "see it. These are their marks against that answer key - "
        f"{smart.recoverable:,} of {smart.scored:,} payments were genuinely recoverable."
    )

    columns = st.columns(4)
    columns[0].metric(
        "Recall",
        f"{smart.recall:.1%}" if smart.recall is not None else "-",
        delta=_delta_points(smart.recall, baseline.recall),
        help="Of the payments that could have been recovered, the share this agent "
        "actually recovered. The one number that separates 'did well' from 'did as well "
        "as was possible'. The Baseline's own recall over the same payments was "
        + (
            f"{baseline.recall:.1%}."
            if baseline.recall is not None
            else "not computable."
        ),
    )
    columns[1].metric(
        "Recoverable, but missed",
        f"{smart.missed_recoverable:,}",
        delta=f"{smart.missed_recoverable - baseline.missed_recoverable:+,} vs baseline",
        delta_color="inverse",
        help="Money that was there to be collected and was not. The breakdown below says "
        f"why. The Baseline missed {baseline.missed_recoverable:,}.",
    )
    columns[2].metric(
        "Attempts on lost causes",
        f"{smart.wasted_attempts:,}",
        delta=f"{smart.wasted_attempts - baseline.wasted_attempts:+,} vs baseline",
        delta_color="inverse",
        help="Retries spent on payments the answer key says were never going to recover. "
        "'Wasted' is hindsight - the agent could not have known, and a reasonable number "
        "of tries on a failure that looks temporary is correct behaviour. What this "
        "measures is how quickly an agent stops paying for a lost cause. The Baseline "
        f"spent {baseline.wasted_attempts:,}.",
    )
    mean_wasted = smart.mean_wasted_attempts
    columns[3].metric(
        "Per lost cause",
        f"{mean_wasted:.2f}" if mean_wasted is not None else "-",
        delta=(
            f"{mean_wasted - baseline.mean_wasted_attempts:+.2f} vs baseline"
            if mean_wasted is not None and baseline.mean_wasted_attempts is not None
            else None
        ),
        delta_color="inverse",
        help="The same waste, per hopeless payment, which is the form worth comparing "
        "between two agents that made different numbers of attempts overall.",
    )

    st.plotly_chart(recall_chart(smart, baseline), use_container_width=True, theme=None)
    st.caption(
        "Both bars are the same length by construction: which payments were recoverable "
        "is a property of the world, fixed before either agent ran."
    )

    # The consistency check. Not a measure of either agent — it is the world checking
    # itself. The simulator only ever approves an attempt its own answer key sanctions,
    # so a non-zero here means the answer key and the fake bank have disagreed and every
    # other number on this page is suspect. Reported either way, rather than assumed.
    impossible = smart.recovered_of_unrecoverable + baseline.recovered_of_unrecoverable
    if impossible:
        st.error(
            f"Consistency check FAILED: {impossible:,} payment(s) were recovered that "
            "the answer key says were impossible. The simulator and its own hidden "
            "truths disagree, so every number on this page is suspect."
        )
    else:
        st.caption(
            "Consistency check passed: neither agent recovered a payment the answer key "
            "called impossible."
        )

    st.markdown("##### Why the recoverable ones got away")
    st.caption(
        "Each agent's own recorded reason for stopping. **These two lists are not "
        "comparable line by line** - the agents stop for different reasons because they "
        "have different mechanisms. The Baseline's journeys can only ever end at its "
        "fixed attempt cap; the Smart Agent's can also end at a circuit-breaker closure "
        "or at the recovery horizon. That is why this is two lists and not one chart."
    )
    missed_columns = st.columns(2)
    for column, agent, scored in (
        (missed_columns[0], AGENT_SMART, smart),
        (missed_columns[1], AGENT_BASELINE, baseline),
    ):
        column.markdown(f"**{AGENT_LABELS[agent]}**")
        frame = _missed_reasons_frame(scored)
        if frame.empty:
            column.caption("No recoverable payment was missed.")
        else:
            column.dataframe(frame, use_container_width=True, hide_index=True)

    _render_classifier_diagnosis(smart.classifier)

    with st.expander("Every diagnostic number, side by side"):
        st.dataframe(
            _truth_table(run.results), use_container_width=True, hide_index=True
        )
        st.caption(
            f'Cells reading "{metrics.NOT_APPLICABLE_NO_CLASSIFIER}" are not zeros - the '
            "Baseline Agent never formed an opinion about why a payment failed, so it "
            "has no diagnosis to be right or wrong about. Rows marked [canary] are 1.0 "
            "by construction and exist to catch a future regression, not to report a "
            'score. "Lost-cause attempts, as a share of all attempts" reads backwards '
            "and should be quoted only alongside the two absolute waste figures above "
            "it: the Smart Agent's share is the higher one precisely because it cut its "
            "total attempts faster than it cut its wasted ones."
        )


def _render_classifier_diagnosis(
    diagnosis: "Optional[metrics.ClassifierDiagnosis]",
) -> None:
    """Operation 7 — what the opening soft/hard call was actually worth.

    **Smart Agent only, and it says so.** The Baseline has no Classifier, so there is no
    second column to draw, no delta to compute and nothing to chart. The absence is
    stated in words rather than left as an empty half of a two-column layout.

    Only one figure here is rendered as a stat tile, and the choice is deliberate: a tile
    reads as a score, and only `attempts_on_a_dead_origin_channel` is one. The two
    1.0-by-construction canaries stay in a plain table, labelled as canaries, where they
    cannot be mistaken for a performance claim.
    """
    st.markdown("##### What the opening diagnosis was worth")
    if diagnosis is None:
        st.caption(
            "Only the Smart Agent forms a diagnosis, so this section has one column. "
            f"For the Baseline Agent this is {metrics.NOT_APPLICABLE_NO_CLASSIFIER}: it "
            "retries without ever forming an opinion about what went wrong, which is not "
            "the same as forming one and getting it wrong."
        )
        return

    st.caption(
        "The Smart Agent opens every journey by reading the decline code and calling the "
        f"failure soft or hard - {diagnosis.calls:,} calls here, "
        f"{diagnosis.soft_calls:,} soft and {diagnosis.hard_calls:,} hard. Only the "
        "Smart Agent does this; for the Baseline Agent every number in this subsection "
        f"is {metrics.NOT_APPLICABLE_NO_CLASSIFIER}."
    )

    dead_retries = diagnosis.attempts_on_a_dead_origin_channel
    st.metric(
        "Retries onto a channel it had already called dead",
        f"{dead_retries:,}",
        help="The behavioural consequence of the diagnosis, and the number this whole "
        "project turns on: hammering a channel the agent's own opening call said was "
        "finished is the single behaviour it exists to prevent. Anything other than 0 is "
        "a defect, not a metric.",
    )
    if dead_retries:
        st.warning(
            f"{dead_retries:,} retries went to a channel the agent had itself diagnosed "
            "as permanently dead. That is a defect, not a result."
        )
    else:
        st.caption(
            "Zero, which is the only correct value here. The agent never spent a retry "
            "on a channel its own diagnosis had written off."
        )

    st.dataframe(
        pd.DataFrame(
            {
                "What was measured": [
                    "Base rate: recoverable somewhere, given a soft call",
                    "Base rate: recoverable elsewhere, given a hard call",
                    "Codes it refused to classify",
                    "Agreement with the simulator's own label [canary]",
                    "Hard-call channel precision [canary]",
                ],
                AGENT_LABELS[AGENT_SMART]: [
                    format_truth_cell(
                        "recoverable_given_soft", diagnosis.recoverable_given_soft
                    ),
                    format_truth_cell(
                        "recoverable_given_hard", diagnosis.recoverable_given_hard
                    ),
                    format_truth_cell(
                        "unrecognized_calls", diagnosis.unrecognized_calls
                    ),
                    format_truth_cell("label_agreement", diagnosis.label_agreement),
                    format_truth_cell(
                        "hard_call_channel_precision",
                        diagnosis.hard_call_channel_precision,
                    ),
                ],
                AGENT_LABELS[AGENT_BASELINE]: [
                    metrics.NOT_APPLICABLE_NO_CLASSIFIER
                ] * 5,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "**The two base rates are not accuracies.** A soft call means 'temporary, worth "
        "another try' - an explicit *maybe*, and the ones that stayed dead are not "
        "mistakes, because no recovery was ever claimed. A hard call is a claim about "
        "the **channel**, not the payment: a hard-declined customer recovered by "
        "switching to their UPI mandate is the classifier being right, not wrong. "
        "**The two [canary] rows are 1.0 by construction** - the simulator picks a "
        "decline code because it has already decided the failure is soft or hard, from "
        "the same registries the classifier reads, so the two cannot disagree unless one "
        "of them is broken. They are kept to catch that breakage and are never quoted as "
        "evidence the classifier works."
    )


def _truth_table(results: BatchResults) -> pd.DataFrame:
    """The full diagnostic comparison, built from the exported frame itself.

    Read off `BatchResults.to_frame()` — the same table the CSV export is generated from —
    rather than from the dataclasses directly, so the screen and the download cannot drift
    apart, and so a metric added to `AgainstTruth` later appears here automatically rather
    than waiting for somebody to remember this function exists.
    """
    frame = results.to_frame()
    rows = frame[frame["section"].isin(("against_truth", "classifier"))]
    values = {
        (row.metric, row.agent): row.value for row in rows.itertuples(index=False)
    }
    section_of = {row.metric: row.section for row in rows.itertuples(index=False)}
    # First-seen order, but with the whole against-truth block ahead of the whole
    # classifier block. Without this the Baseline's one extra terminal-reason row — a tag
    # only it can emit — lands after the classifier rows, stranded far from the Smart
    # Agent's equivalent row it is meant to be read against.
    ordered = sorted(
        dict.fromkeys(rows["metric"]),
        key=lambda metric: section_of[metric] == "classifier",
    )

    def cell(metric: str, agent: str) -> str:
        if (metric, agent) in values:
            return format_truth_cell(metric, values[(metric, agent)])
        # Absent — and the two ways a cell can be absent do not mean the same thing.
        if section_of[metric] == "classifier":
            return metrics.NOT_APPLICABLE_NO_CLASSIFIER
        if metric.startswith("missed_by_terminal_reason."):
            return NOT_RECORDED_BY_THIS_AGENT
        return "none"

    return pd.DataFrame(
        {
            "Metric": [_truth_label(metric) for metric in ordered],
            AGENT_LABELS[AGENT_SMART]: [cell(m, AGENT_SMART) for m in ordered],
            AGENT_LABELS[AGENT_BASELINE]: [cell(m, AGENT_BASELINE) for m in ordered],
        }
    )


def _truth_label(metric: str) -> str:
    """A readable row name, or the raw metric name if nobody wrote one.

    Terminal-reason rows are named by the Audit Trail's own English for the tag, so the
    table says the same thing about an ending that the narration does.
    """
    prefix = "missed_by_terminal_reason."
    if metric.startswith(prefix):
        return f"Missed because {reason_phrase(metric[len(prefix):])}"
    return TRUTH_METRIC_LABELS.get(metric, metric)


# ---------------------------------------------------------------------------
# The export (C4, operation 4)
# ---------------------------------------------------------------------------
#
# "Provide the audit trail export (CSV/JSON) as a downloadable artifact from this layer."
#
# The point of this section is what happens *after* the demo. A judge who watched the feed
# scroll past saw the reasoning once; IDEA.md §10 makes the export a graded requirement
# precisely so there is something concrete to open and inspect afterwards, outside the
# room. Everything below is text already sitting in memory — `AuditTrail.to_csv()` and
# `BatchResults.to_json()` return strings for exactly this reason (C1 operation 4, C3
# operation 8) — so nothing is written to the machine running the demo.
#
# **Provenance lives in the filename.** This is the detail worth defending. Phase 11 task 3
# built the whole `run_key` fingerprint on the argument that a number quoted without the
# world it came from is not checkable — and a file called `audit_trail.csv`, sitting in a
# downloads folder a week later, is exactly that number. So every file this page hands out
# is named `<what>_<run key>_seed<n>.<ext>`, and the key is a fingerprint of the entire
# configuration, not just the seed: the same seed at a different failure rate is a
# different world and gets a different name.


DOWNLOAD_MIME = {"csv": "text/csv", "json": "application/json"}


def export_filename(stem: str, run_key: str, seed: int, extension: str) -> str:
    """`audit_trail_2de739a8279f_seed7.csv` — what the file is, and which world it is of.

    Both halves are needed and neither is redundant. The seed is what a person says out
    loud and types back into the sidebar; the key is what actually identifies the run,
    since the same seed under a different failure rate or agent configuration is a
    different experiment entirely.
    """
    return f"{stem}_{run_key}_seed{seed}.{extension}"


@st.cache_data(show_spinner=False)
def _trail_export(_trail: AuditTrail, run_key: str, extension: str) -> str:
    """The audit trail as downloadable text, computed once per run and format.

    **Why this is cached, and why the cache key is what it is.** Streamlit re-executes this
    whole script on every interaction — every dropdown change in the drill-down below —
    and a demo-scale trail is thousands of entries. Rendering both export formats on every
    keystroke is work nobody asked for.

    `_trail` is underscore-prefixed so Streamlit does not try to hash it (an `AuditTrail`
    is not hashable, and hashing a few thousand entries to save rendering them would be
    self-defeating). That means the cache is keyed on `run_key` and `extension` **only**,
    which is correct precisely because the key is a fingerprint of the run — but it also
    means passing a *stale or missing* key here would silently serve one run's audit trail
    under another run's name. That is the failure this signature exists to prevent, and
    `tests/test_dashboard.py` runs two different worlds in one session to prove it does.
    """
    return _trail.to_csv() if extension == "csv" else _trail.to_json()


@st.cache_data(show_spinner=False)
def _results_export(_results: BatchResults, run_key: str, extension: str) -> str:
    """The scored metrics as downloadable text. Cached on the same terms as `_trail_export`.

    This is the file that carries the configuration *inside* it as well as in its name:
    `BatchResults` stores the simulator, orchestrator and baseline configs alongside the
    numbers, so somebody handed this JSON can rebuild the run it describes with
    `metrics.run_evaluation()` and check that the key comes back the same.
    """
    return _results.to_csv() if extension == "csv" else _results.to_json()


def render_exports(run: CompletedRun) -> None:
    """The download controls (ARCHITECTURE.md C4, operation 4).

    Four files, in two pairs. The audit trail is **every decision both agents made, with
    the plain-English sentence beside the facts that drove it** — the CSV opens in a
    spreadsheet, the JSON is one object per decision. The results are the scored
    comparison — the CSV is one metric per row for stacking several runs together, the
    JSON is the whole nested object including the configuration that produced it.

    **Two absences are handled here, and they are different from each other.** A run in
    which nothing failed has an empty trail *and* no scored results; a run that failed but
    could not be scored cannot happen (the same condition produces both), but the two are
    checked separately anyway, because a control offering a file that does not exist is
    how a demo meets a stack trace. Neither is padded into an empty file: an export button
    that hands over a zero-row CSV looks like a broken export rather than an empty world.
    """
    st.subheader("Take it away")

    if not len(run.trail):
        st.info(
            "This run produced no decisions to export - nothing failed, so neither agent "
            "ever had to decide anything."
        )
        return

    seed = run.settings.simulator.seed
    st.caption(
        f"Every file is named for the run it came from - `{run.key}`, seed {seed} - "
        "because a number quoted without the world it came from cannot be checked. The "
        "key is a fingerprint of the whole configuration, not just the seed."
    )

    columns = st.columns(4)
    columns[0].download_button(
        "Audit trail (CSV)",
        data=_trail_export(run.trail, run.key, "csv"),
        file_name=export_filename("audit_trail", run.key, seed, "csv"),
        mime=DOWNLOAD_MIME["csv"],
        use_container_width=True,
        help=f"All {len(run.trail):,} decisions both agents made, one per row, each with "
        "the plain-English sentence explaining it. Opens in a spreadsheet.",
    )
    columns[1].download_button(
        "Audit trail (JSON)",
        data=_trail_export(run.trail, run.key, "json"),
        file_name=export_filename("audit_trail", run.key, seed, "json"),
        mime=DOWNLOAD_MIME["json"],
        use_container_width=True,
        help="The same log as one object per decision, for reading with code.",
    )

    if run.results is None:
        # The recurring shape of this phase's one real bug: a view that assumed a score
        # exists. It does not here, and the honest move is to say so rather than to hand
        # over a file of nulls.
        columns[2].caption(
            "No scored results to export: nothing failed in this world, so there is no "
            "recovery rate to report."
        )
        return

    columns[2].download_button(
        "Results (JSON)",
        data=_results_export(run.results, run.key, "json"),
        file_name=export_filename("results", run.key, seed, "json"),
        mime=DOWNLOAD_MIME["json"],
        use_container_width=True,
        help="Every number on this page, plus the exact configuration that produced it - "
        "enough to rebuild this run from scratch and check the key comes back the same.",
    )
    columns[3].download_button(
        "Results (CSV)",
        data=_results_export(run.results, run.key, "csv"),
        file_name=export_filename("results", run.key, seed, "csv"),
        mime=DOWNLOAD_MIME["csv"],
        use_container_width=True,
        help="The same numbers as one metric per row, which is the shape that stacks: "
        "several runs' files concatenate into one table without reshaping.",
    )


def render_drilldown(run: CompletedRun) -> None:
    """One transaction's full story, top to bottom (C1 operation 3, surfaced as the
    drill-down C4 operation 1 leads into).

    The live feed shows a moving window; this is where a judge who saw a sentence go past
    can stop and read that transaction's whole reasoning chain. Both agents' journeys over
    the same billing event are offered side by side, because the interesting question is
    almost never "what did the Smart Agent do" but "what did it do *differently*".
    """
    st.subheader("Read one transaction end to end")

    smart_ids = [r.transaction.transaction_id for r in run.paired.smart]
    if not smart_ids:
        st.info("This run produced no failed billing events — nothing to narrate.")
        return

    chosen = st.selectbox(
        "Transaction",
        smart_ids,
        help="Every failed billing event the Smart Agent handled, in the order it ran.",
    )

    smart_column, baseline_column = st.columns(2)
    smart_result = next(
        r for r in run.paired.smart if r.transaction.transaction_id == chosen
    )
    billing_event_id = smart_result.transaction.billing_event_id
    baseline_result = next(
        (r for r in run.paired.baseline if r.transaction.billing_event_id == billing_event_id),
        None,
    )

    _render_journey(smart_column, AGENT_SMART, smart_result, run.trail)
    if baseline_result is None:
        # Should not happen — both agents run over the same filtered events — but a
        # missing counterpart is reported rather than silently rendering an empty column.
        baseline_column.warning("No Baseline journey recorded for this billing event.")
    else:
        _render_journey(baseline_column, AGENT_BASELINE, baseline_result, run.trail)


def _render_journey(
    column: st.delta_generator.DeltaGenerator,
    agent: str,
    result: AgentResult,
    trail: AuditTrail,
) -> None:
    """One agent's half of the drill-down: how the journey ended, then every sentence."""
    status = result.transaction.status
    column.markdown(f"#### {AGENT_LABELS[agent]}")
    column.markdown(
        f"**{len(result.transaction.attempts)} attempts** · "
        f"{STATUS_LABELS.get(status, status.value)}"
    )
    for line in trail.narrate(result.transaction.transaction_id):
        column.markdown(line)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("AI Revenue Recovery Agent")
    st.markdown(
        "A failed subscription payment is not one problem — it is several, and they need "
        "different answers. This agent diagnoses each decline, chooses **how** and "
        "**when** to retry it, and knows when to stop. Every decision below is one it "
        "wrote down as it made it."
    )

    settings = sidebar_settings()
    if st.sidebar.button("Run batch", type="primary", use_container_width=True):
        st.session_state["run"] = run_live(settings)

    run: Optional[CompletedRun] = st.session_state.get("run")

    if run is None:
        st.info(
            "Choose a seed and population in the sidebar, then press **Run batch** to "
            "watch both agents work through the same set of failed payments."
        )
        return

    st.divider()
    render_head_to_head(run)

    st.divider()
    render_against_truth(run)

    st.divider()
    render_exports(run)

    st.divider()
    render_drilldown(run)


main()
