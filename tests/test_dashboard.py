"""Tests for the Dashboard (C4).

**Why this file exists**, given that DESIGN.md §5 scopes `pytest` to the Classifier,
Bandit and Circuit Breaker and calls the dashboard "presentation code":

Presentation is exactly where a true number becomes a false claim. Every honesty rule the
Metrics engine (C3) enforces lives in a *value* — `channel_closures` is `None`, not `0`,
for an agent that has no circuit breaker. A dashboard can obey that rule at the data layer
and still break it on screen, because charting libraries render a missing value as zero by
default and nobody looking at the picture can tell the difference. `tests/test_metrics.py`
guards the number; nothing guards the pixel.

The properties below are the ones where this page could show something *plausible and
wrong* while every component beneath it behaves correctly:

1. **A closure count of 0 shown for the Baseline Agent**, or a delta computed against one.
   The single most misleading thing this page could display: it invites a reader to
   compare 43 against 0 as though both agents had counted, when only one of them has the
   mechanism at all.
2. **A metric only one agent has, plotted on a chart showing both.** A bar chart has no
   way to draw "absent" — it draws nothing, which reads as zero.
3. **A metric silently dropped from the comparison table** because nobody wrote a display
   label for it. A missing row is invisible in a way a wrong row is not.
4. **The page disagreeing with the export.** The tiles on screen and the JSON a judge
   downloads must be two views of one run, not two computations that happen to agree
   today.

Same precedent and the same reasoning as `tests/test_audit_trail.py` and
`tests/test_metrics.py`: a defect in what a *display* claims is invisible to tests of the
values it displays.

These run through `streamlit.testing.v1.AppTest`, which executes the real script the way
Streamlit does — so what is asserted here is what a viewer would actually see.
"""

import json
from pathlib import Path

import pytest

from recovery_agent.metrics import NOT_APPLICABLE_NO_BREAKER

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit is required for the dashboard tests"
).AppTest

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"

SMART_LABEL = "Smart Agent"
BASELINE_LABEL = "Baseline Agent"


@pytest.fixture(scope="module")
def app():
    """One completed batch, shared across the module.

    Module-scoped because a run is the expensive part and none of these tests mutate it —
    they read what the finished page rendered. The population is small enough to run end
    to end in a couple of seconds, but not the smallest the sidebar allows: at 20
    customers every failure happens to recover, so the outcome chart would be asserted
    against a world with two empty segments and the interesting cases would go unchecked.
    """
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.slider[0].set_value(60)   # customers
    at.sidebar.slider[2].set_value(0.0)  # narration pause: display only
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    return at


def _outcome_chart_spec(app):
    """The outcome chart's figure, as it was actually handed to the browser.

    Plotly charts have no typed accessor in this version of `AppTest`, so the element is
    fetched by name and its spec read back from the protobuf. That is the honest place to
    assert from anyway: it is the figure that was sent, not the arguments we passed to the
    function that built it.
    """
    charts = app.get("plotly_chart")
    assert charts, "the outcome chart was not rendered"
    return json.loads(charts[0].proto.spec)


def _comparison_table(app):
    """The 'Every metric, side by side' table as it was handed to Streamlit."""
    assert app.dataframe, "the comparison table was not rendered"
    return app.dataframe[0].value


def test_baseline_closures_are_never_shown_as_zero(app):
    """The honesty property, at the pixel: an agent with no circuit breaker reports no
    closure count anywhere on the page — not in the tile, not in the table."""
    table = _comparison_table(app)

    closures = table.loc[table["Metric"] == "Channels permanently retired"]
    assert len(closures) == 1, "the closure row is missing from the comparison table"
    cell = closures[BASELINE_LABEL].iloc[0]
    assert cell == NOT_APPLICABLE_NO_BREAKER
    assert "0" not in str(cell)

    # Same rule for the two metrics derived from the same absent mechanism.
    for metric in ("Payments that lost a channel", "Closures by channel"):
        row = table.loc[table["Metric"] == metric]
        assert len(row) == 1, f"{metric!r} is missing from the comparison table"
        assert row[BASELINE_LABEL].iloc[0] == NOT_APPLICABLE_NO_BREAKER


def test_closure_tile_carries_no_delta(app):
    """The closure tile must not show a change 'vs baseline'.

    There is no baseline number to subtract. A delta here would be arithmetic performed
    against a value that does not exist, presented as a comparison.
    """
    tiles = {m.label: m for m in app.metric}
    tile = tiles.get("Channels permanently retired")
    assert tile is not None, "the closure tile is missing"
    assert not tile.delta, f"closure tile showed a delta: {tile.delta!r}"

    # And the absence is stated in words rather than left as a blank the reader fills in.
    assert any(
        NOT_APPLICABLE_NO_BREAKER in caption.value for caption in app.caption
    ), "nothing on the page says why the Baseline has no closure count"


def test_no_single_agent_metric_is_plotted(app):
    """A metric only one agent has never goes on a chart that shows both.

    A chart cannot draw "absent" — it draws nothing, and nothing reads as zero. Every
    series on the outcome chart must therefore be a metric both agents genuinely produced.
    """
    spec = _outcome_chart_spec(app)
    plotted = {trace["name"] for trace in spec["data"]}
    assert plotted == {"Recovered", "Escalated to a human", "Abandoned"}

    for trace in spec["data"]:
        assert len(trace["x"]) == 2, "a series is missing one of the two agents"
        assert all(value is not None for value in trace["x"]), (
            f"series {trace['name']!r} has a missing value, which a bar chart draws as 0"
        )
        # Direct labels are the relief for the low-contrast palette slot; without them
        # identity on this chart would rest on colour alone.
        assert len(trace["text"]) == 2

    # And the two bars are the same length, because both agents were handed the same
    # payments. A chart where they were not would be comparing two different experiments.
    totals = [sum(trace["x"][i] for trace in spec["data"]) for i in (0, 1)]
    assert totals[0] == totals[1]


def test_every_metric_reaches_the_table(app):
    """No row is dropped on its way to the screen.

    The display-label map is allowed to *rename* metrics and nothing else. A metric added
    to `AgentSummary` later, with no label written for it, must still appear — under its
    raw name if necessary — rather than silently vanishing from the comparison.
    """
    from recovery_agent.metrics import head_to_head

    run = app.session_state["run"]
    expected = head_to_head(run.paired).to_frame()
    table = _comparison_table(app)

    assert len(table) == len(expected), (
        f"comparison table shows {len(table)} of {len(expected)} metrics"
    )
    assert not table["Metric"].duplicated().any()


def test_page_and_export_describe_the_same_run(app):
    """What is on screen and what a judge downloads must be one run, not two.

    The tiles are checked against the same `BatchResults` the export is built from, so a
    view that recomputed its own numbers — or that scored a second, freshly-run batch —
    would fail here.
    """
    run = app.session_state["run"]
    head = run.results.head_to_head
    tiles = {m.label: m for m in app.metric}

    assert tiles["Recovery rate"].value == f"{head.smart.recovery_rate:.1%}"
    assert tiles["Retry attempts used"].value == f"{head.smart.total_attempts:,}"
    assert tiles["Escalated to a human"].value == f"{head.smart.escalated_to_human:,}"

    # The delta is the metrics engine's own derived figure, not one the page re-derives.
    assert f"{head.recovery_rate_gain_points:+.1f}" in tiles["Recovery rate"].delta

    # And the run is scored from the batch that was actually narrated: the key is a
    # fingerprint of the configuration, so a second run of a different world would differ.
    assert run.results.key in run.results.label


def test_rates_are_shown_as_percentages(app):
    """A recovery rate rendered as `0.6835` is not wrong, but nobody reads it. The rate
    rows are formatted as percentages; the count rows are not."""
    table = _comparison_table(app)
    rate_row = table.loc[table["Metric"] == "Recovery rate"]
    assert rate_row[SMART_LABEL].iloc[0].endswith("%")

    count_row = table.loc[table["Metric"] == "Failed payments handled"]
    assert not count_row[SMART_LABEL].iloc[0].endswith("%")


def test_both_agents_faced_the_same_payments(app):
    """The premise the whole comparison rests on, asserted where a viewer would read it.

    If the two agents were handed different numbers of payments, every rate on this page
    would be computed over a different denominator and the side-by-side would be
    meaningless — while still looking entirely normal.
    """
    run = app.session_state["run"]
    head = run.results.head_to_head
    assert head.smart.transactions == head.baseline.transactions
    assert head.smart.transactions > 0

    table = _comparison_table(app)
    handled = table.loc[table["Metric"] == "Failed payments handled"]
    assert handled[SMART_LABEL].iloc[0] == handled[BASELINE_LABEL].iloc[0]


def test_a_world_with_no_failures_does_not_crash_or_invent_a_rate():
    """A small population at a low failure rate can produce a world in which nothing
    failed. That is reachable straight from the sidebar, and it is not a batch with a
    recovery rate of 0% — there is no denominator, so there is no rate.

    The page must say so and stay up. Scoring such a run is what the Metrics engine
    explicitly refuses to do, and a dashboard that called it anyway would greet a judge
    with a stack trace mid-demo.
    """
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.number_input[0].set_value(3)      # a seed whose small world has no failures
    at.sidebar.slider[0].set_value(20)           # customers
    at.sidebar.slider[1].set_value(0.05)         # base failure rate
    at.sidebar.slider[2].set_value(0.0)
    at.sidebar.button[0].click().run()

    assert not at.exception, at.exception
    run = at.session_state["run"]
    assert run.paired.smart == []
    assert run.results is None, "an empty batch must not be scored"

    said_so = [i.value for i in at.info if "nothing to recover" in i.value]
    assert said_so, "the page did not explain why there is no comparison"
    assert "0%" not in said_so[0], "an empty batch must not be reported as a 0% rate"
