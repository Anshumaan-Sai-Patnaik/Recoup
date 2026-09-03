"""C4 — Dashboard / Reporting Layer (Streamlit entrypoint).

    streamlit run dashboard/app.py

**Presentation only.** ARCHITECTURE.md C4 is explicit that this layer "owns no decision
logic of its own" and is a leaf node: nothing it produces feeds back into any other
component. Every number and every sentence on this page was computed by something else
and is displayed here verbatim. If a fact is not on a result object, this file does not
invent it.

Currently covers C4 operations 1 and 2. **Operation 1** is the live narration feed, which
IDEA.md §10 names as the single most important live-demo beat: instead of a black box,
judges watch the agent narrate its own reasoning in real time. **Operation 2** is the
head-to-head Baseline-vs-Smart comparison once the batch completes — the "that comparison
*is* the pitch" beat from IDEA.md §11. Operations 3-5 (the against-hidden-truth view, the
export button and the fuller run configuration) arrive in the remaining Phase 12 tasks;
each is a view appended below rather than a replacement for what is here.

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


def sidebar_settings() -> RunSettings:
    """The run controls (part of C4 operation 5 — enough of it for operation 1 to have
    something to narrate; the fuller configuration surface is a later Phase 12 task).

    Only the three simulator fields that change what a demo *shows* are exposed. Both
    agent configurations are left at their defaults here on purpose: they are what every
    number recorded in `notes/TRACKER.md` was produced under, and a judge who nudges a
    slider should still be looking at the same experiment.
    """
    st.sidebar.header("Run a batch")

    seed = st.sidebar.number_input(
        "Seed",
        min_value=0,
        max_value=999_999,
        value=11,
        step=1,
        help=(
            "The world. The same seed always produces the same customers, the same "
            "billing events and the same hidden truths — so both agents can be compared "
            "on identical ground, and any run here can be reproduced later."
        ),
    )
    num_customers = st.sidebar.slider(
        "Customers",
        min_value=20,
        max_value=1000,
        value=200,
        step=20,
        help="How large a population to generate. Only the failed billing events are retried.",
    )
    base_failure_rate = st.sidebar.slider(
        "Base failure rate",
        min_value=0.05,
        max_value=0.50,
        value=0.25,
        step=0.05,
        help=(
            "How often a billing event fails in the first place. ~8% is the commonly "
            "cited industry ballpark; the demo default is raised to 25% so a modest "
            "population still produces enough failures to compare."
        ),
    )

    st.sidebar.divider()
    narration_delay = st.sidebar.slider(
        "Narration pause (seconds per transaction)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.05,
        help=(
            "Display speed only. Simulated time is instant, so the whole batch finishes "
            "in about a second — too fast to read. This pauses the *screen* between "
            "transactions. It changes nothing about the run: same seed, same decisions, "
            "same results, with or without it."
        ),
    )

    return RunSettings(
        simulator=SimulatorConfig(
            seed=int(seed),
            num_customers=int(num_customers),
            base_failure_rate=float(base_failure_rate),
        ),
        narration_delay_seconds=float(narration_delay),
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
    return CompletedRun(paired=paired, trail=trail, results=results)


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
    counts are drawn directly on the segments and the full table is one click away, so
    nothing here depends on telling two fills apart.
    """
    outcomes = [
        ("Recovered", SERIES_RECOVERED, "recovered"),
        ("Escalated to a human", SERIES_ESCALATED, "escalated_to_human"),
        ("Abandoned", SERIES_ABANDONED, "abandoned"),
    ]
    agents = [
        (AGENT_LABELS[AGENT_BASELINE], head.baseline),
        (AGENT_LABELS[AGENT_SMART], head.smart),
    ]

    # Both agents handled the same payments, so a bar's total is the denominator for
    # deciding which of its segments are wide enough to carry a label.
    totals = [
        sum(getattr(summary, field) for _, _, field in outcomes)
        for _, summary in agents
    ]

    figure = go.Figure()
    for name, colour, field_name in outcomes:
        counts = [getattr(summary, field_name) for _, summary in agents]
        figure.add_bar(
            y=[label for label, _ in agents],
            x=counts,
            name=name,
            orientation="h",
            marker_color=colour,
            # A 2px surface-coloured gap between adjacent fills, so two segments never
            # merge into one block at a glance.
            marker_line_color=CHART_SURFACE,
            marker_line_width=2,
            # Direct labels, but only where one fits: a number inside a segment narrower
            # than its own digits would be clipped, and labelling every mark regardless is
            # how a chart becomes noise. The rest stay on hover and in the table below,
            # which is also the relief the low-contrast slot needs.
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
            hovertemplate="%{y}<br>" + name + ": %{x:,} payments<extra></extra>",
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
    # Recessive axes: the marks carry their own numbers, so the scale underneath them is
    # scaffolding and should not compete with the data.
    figure.update_xaxes(visible=False)
    figure.update_yaxes(
        showgrid=False,
        zeroline=False,
        ticksuffix="  ",
        tickfont=dict(color=TEXT_SECONDARY),
    )
    return figure


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
    render_drilldown(run)


main()
