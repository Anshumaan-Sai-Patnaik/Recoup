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

from recovery_agent.metrics import (
    NOT_APPLICABLE_NO_BREAKER,
    NOT_APPLICABLE_NO_CLASSIFIER,
)

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
    at.sidebar.slider("customers").set_value(60)
    at.sidebar.slider("narration_delay").set_value(0.0)  # display only
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    return at


@pytest.fixture(scope="module")
def app_with_divergent_endings():
    """A second, larger completed batch — the one test below needs a world where *both*
    agents leave something behind, and for different reasons.

    At 60 customers the Smart Agent recovers everything the answer key says was
    recoverable, so only the Baseline contributes a stopping reason and the test it feeds
    would pass vacuously. This is the smallest round population where the Smart Agent also
    runs out of channels on at least one payment, which is the situation the test exists to
    check the rendering of. Kept separate from `app` rather than replacing it, because the
    other tests here are calibrated against that smaller world.
    """
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.slider("customers").set_value(200)
    at.sidebar.slider("narration_delay").set_value(0.0)  # display only
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
    at.sidebar.number_input("seed").set_value(3)   # a small world with no failures
    at.sidebar.slider("customers").set_value(20)
    at.sidebar.slider("failure_rate").set_value(0.05)
    at.sidebar.slider("narration_delay").set_value(0.0)
    at.sidebar.button[0].click().run()

    assert not at.exception, at.exception
    run = at.session_state["run"]
    assert run.paired.smart == []
    assert run.results is None, "an empty batch must not be scored"

    said_so = [i.value for i in at.info if "nothing to recover" in i.value]
    assert said_so, "the page did not explain why there is no comparison"
    assert "0%" not in said_so[0], "an empty batch must not be reported as a 0% rate"

    # Every view added after the head-to-head has to survive the same world. The
    # against-truth view reads the same `results` object and must decline in the same
    # way rather than scoring an empty batch or rendering an empty chart.
    scored_nothing = [i.value for i in at.info if "hidden truth" in i.value]
    assert scored_nothing, "the against-truth view said nothing about an empty batch"
    assert not at.get("plotly_chart"), "a chart was drawn for a batch with no payments"

    # An export control offering a file that does not exist is how a demo meets a stack
    # trace. There is nothing to hand over here, and the page says so instead of
    # producing a zero-row CSV that looks like a broken export.
    assert not at.get("download_button"), "a download was offered for a run with no decisions"
    assert [c for c in at.info if "no decisions to export" in c.value], (
        "the page did not explain why there is nothing to download"
    )


# ---------------------------------------------------------------------------
# The against-hidden-truth view (C4, operation 3)
#
# Same rule as above, in a place where it is easier to break and more tempting to break:
# these numbers read as "the agent was right", so a display that quietly rounds an
# absence up to a score, or puts a 1.0-by-construction canary on a stat tile, would be
# making a performance claim the Metrics engine explicitly refuses to make.
# ---------------------------------------------------------------------------


NOT_RECORDED_BY_THIS_AGENT = "not a reason this agent can record"
"""Repeated here as a literal rather than imported from `dashboard/app.py`.

Importing the app module would execute it outside a Streamlit run; more to the point, a
test that reads its expected value out of the code under test can only ever agree with
it. The words are pinned here on purpose.
"""


def _table_containing(app, metric: str):
    """The rendered table that carries `metric` in its Metric column.

    Fetched by content rather than by index: the page renders several tables, and pinning
    one by position would turn "a table was added above this one" into a silent mis-assert
    rather than a failure.
    """
    for element in app.dataframe:
        frame = element.value
        if "Metric" in frame.columns and (frame["Metric"] == metric).any():
            return frame
    raise AssertionError(f"no rendered table contains the metric {metric!r}")


def _diagnostic_table(app):
    """The 'Every diagnostic number, side by side' table."""
    return _table_containing(app, "Recall (share of the possible, captured)")


def _chart_with_series(app, names: set):
    """The rendered figure whose series are exactly `names`."""
    for element in app.get("plotly_chart"):
        spec = json.loads(element.proto.spec)
        if {trace["name"] for trace in spec["data"]} == names:
            return spec
    raise AssertionError(f"no chart was rendered with the series {sorted(names)}")


def test_classifier_numbers_are_never_shown_for_the_baseline(app):
    """The task-2 honesty rule, applied to a second absent mechanism.

    `AgainstTruth.classifier` is `None` for the Baseline Agent because it never forms a
    diagnosis — not because it formed one and scored zero. Every operation-7 row must say
    so in words, and none of them may render as a number the reader can compare against
    the Smart Agent's.
    """
    table = _diagnostic_table(app)
    classifier_rows = [
        "Opening diagnoses recorded",
        "Called soft (temporary)",
        "Called hard (this channel is dead)",
        "Refused to classify (code not in the registry)",
        "Agreement with the simulator's own label [canary]",
        "Base rate: recoverable, given a soft call",
        "Hard-call channel precision [canary]",
        "Retries onto a channel it had called dead",
    ]
    for metric in classifier_rows:
        row = table.loc[table["Metric"] == metric]
        assert len(row) == 1, f"{metric!r} is missing from the diagnostic table"
        cell = str(row[BASELINE_LABEL].iloc[0])
        assert cell == NOT_APPLICABLE_NO_CLASSIFIER, (
            f"{metric!r} showed the Baseline {cell!r}, which invites a comparison "
            "against an agent that never made the call"
        )
        assert not any(character.isdigit() for character in cell)


def test_a_reason_one_agent_cannot_record_is_not_shown_as_zero(
    app_with_divergent_endings,
):
    """The two agents stop for different reasons, because they have different mechanisms.

    The Baseline can only ever end a journey at its fixed attempt cap; the Smart Agent can
    also end at a circuit-breaker closure or at the recovery horizon. A `0` in the other
    agent's column would read as "this agent avoided that ending" when the truth is that
    the ending does not exist for it — the same defect as a closure count of zero, in a
    place where it is much easier to miss.
    """
    table = _diagnostic_table(app_with_divergent_endings)
    reason_rows = table.loc[table["Metric"].str.startswith("Missed because")]
    assert len(reason_rows) >= 2, (
        "expected each agent to have contributed at least one distinct stopping reason"
    )

    crossed = [
        (row["Metric"], row[column])
        for _, row in reason_rows.iterrows()
        for column in (SMART_LABEL, BASELINE_LABEL)
        if str(row[column]) == NOT_RECORDED_BY_THIS_AGENT
    ]
    assert crossed, (
        "no reason was absent for either agent, so this test proved nothing — pick a "
        "world where the two agents' journeys end differently"
    )
    for metric, cell in crossed:
        assert "0" not in str(cell), f"{metric!r} rendered an absent reason as a number"


def test_the_recall_chart_plots_only_metrics_both_agents_have(app):
    """A chart cannot draw "absent". The recall chart therefore carries only recovered and
    missed — both of which every agent genuinely produces — and never the Classifier's
    figures, which exist for one agent only.

    Its two bars must also be the same length: which payments were recoverable is a
    property of the world, fixed before either agent ran, so bars of different lengths
    would mean the two agents were scored against two different answer keys.
    """
    spec = _chart_with_series(app, {"Recovered", "Missed"})
    for trace in spec["data"]:
        assert len(trace["x"]) == 2, "a series is missing one of the two agents"
        assert all(value is not None for value in trace["x"])
        assert len(trace["text"]) == 2

    totals = [sum(trace["x"][index] for trace in spec["data"]) for index in (0, 1)]
    assert totals[0] == totals[1], (
        "the two agents were scored against different sets of recoverable payments"
    )

    # And no chart anywhere on the page carries a single-agent series.
    for element in app.get("plotly_chart"):
        chart = json.loads(element.proto.spec)
        for trace in chart["data"]:
            assert len(trace["x"]) == 2


def test_the_by_construction_canaries_are_not_presented_as_scores(app):
    """`label_agreement` and `hard_call_channel_precision` are 1.0 by construction: the
    simulator picks a decline code *because* it has already decided the failure is soft or
    hard, from the same registries the classifier reads.

    A stat tile reading "100%" is what gets screenshotted and quoted as accuracy. These
    two live in a plain table instead, labelled as canaries in the row name itself — where
    a cropped screenshot cannot separate the number from its caveat.
    """
    tile_labels = [tile.label for tile in app.metric]
    for banned in ("agreement", "precision", "accuracy", "canary"):
        assert not any(banned in label.lower() for label in tile_labels), (
            f"a by-construction figure was promoted to a stat tile: {tile_labels}"
        )

    table = _diagnostic_table(app)
    canaries = table.loc[table["Metric"].str.contains(r"\[canary\]")]
    assert len(canaries) >= 2, "the canary rows lost their label"


def test_the_diagnostic_view_matches_the_results_object(app):
    """The against-truth tiles are read off the same `BatchResults` the export is built
    from, so a view that recomputed its own numbers would fail here."""
    run = app.session_state["run"]
    smart = run.results.smart_vs_truth
    baseline = run.results.baseline_vs_truth
    tiles = {tile.label: tile for tile in app.metric}

    assert tiles["Recall"].value == f"{smart.recall:.1%}"
    assert tiles["Recoverable, but missed"].value == f"{smart.missed_recoverable:,}"
    assert tiles["Attempts on lost causes"].value == f"{smart.wasted_attempts:,}"
    assert f"{(smart.recall - baseline.recall) * 100:+.1f}" in tiles["Recall"].delta

    # The consistency check the Metrics engine asks for: a recovery the answer key says
    # was impossible means the fake bank and its own hidden truths have disagreed.
    assert smart.recovered_of_unrecoverable == 0
    assert baseline.recovered_of_unrecoverable == 0


def test_every_diagnostic_number_reaches_the_table(app):
    """No diagnostic metric is dropped on its way to the screen.

    Built from `BatchResults.to_frame()` — the same table the CSV export is generated
    from — so a metric added to `AgainstTruth` later shows up here automatically. This
    test is what stops it from being quietly filtered out instead.
    """
    run = app.session_state["run"]
    frame = run.results.to_frame()
    expected = frame[frame["section"].isin(("against_truth", "classifier"))]
    table = _diagnostic_table(app)

    assert len(table) == expected["metric"].nunique()
    assert not table["Metric"].duplicated().any()


# ---------------------------------------------------------------------------
# The export (C4, operation 4)
#
# What can and cannot be asserted through `AppTest`, stated once so the split below does
# not look arbitrary: a download button's *content* never reaches the protobuf — Streamlit
# stores the bytes in its media file manager and puts a content-hashed URL on the element.
# The filename is not on the element at all. So the export is tested from two directions:
# the naming and text-building rules as plain functions, and everything a viewer actually
# interacts with through the rendered page. The content hash in the URL turns out to be the
# sharpest tool available for the one bug that would really hurt here — a cached file from
# a previous run being served under the current run's name.
# ---------------------------------------------------------------------------


def _load_app_module():
    """`dashboard/app.py` imported as a plain module, for its pure helpers.

    Registered in `sys.modules` before execution because `@dataclass` resolves annotations
    through `sys.modules[cls.__module__]` and fails on a module that is not there yet.
    Importing it executes `main()` in Streamlit's bare mode, which finds no session state
    and returns immediately — noisy in the log, harmless, and the only way to reach
    functions whose output never surfaces through a protobuf.
    """
    import importlib.util
    import sys

    name = "dashboard_app_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(APP))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app_module():
    """The same module, as a fixture for the tests that only need its pure functions."""
    return _load_app_module()


def test_exported_filenames_carry_the_world_they_came_from(app_module):
    """A file called `audit_trail.csv` in a downloads folder a week later says nothing
    about which run produced it — which is exactly the unciteable number Phase 11 task 3
    built the run key to prevent.

    Both halves have to be in the name. The seed is what a person says out loud and types
    back into the sidebar; the key is what actually identifies the experiment, since the
    same seed at a different failure rate is a different world.
    """
    name = app_module.export_filename("audit_trail", "2de739a8279f", 7, "csv")
    assert name == "audit_trail_2de739a8279f_seed7.csv"
    assert "2de739a8279f" in name and "seed7" in name


def test_the_export_is_keyed_on_the_run_it_describes(app_module, app):
    """The cache behind the download buttons is keyed on the run key, and the run key is a
    fingerprint of the whole configuration — so two different worlds cannot collide.

    This is the half of the staleness bug that can be checked directly; the other half,
    that the page actually passes the *current* run's key, is checked end to end below.
    """
    run = app.session_state["run"]
    assert run.key, "the run carries no fingerprint to name its files with"
    assert run.key == run.results.key, (
        "the key the export names files with and the key inside the exported results "
        "disagree, so a file and its contents would cite two different runs"
    )

    trail_csv = app_module._trail_export(run.trail, run.key, "csv")
    trail_json = app_module._trail_export(run.trail, run.key, "json")
    assert trail_csv and trail_json and trail_csv != trail_json
    # The English sentence is in the export, not only on the screen: the file has to be
    # readable on its own by somebody who never saw the live feed.
    assert "sentence" in trail_csv.splitlines()[0]
    assert len(trail_csv.splitlines()) == len(run.trail) + 1  # + the header row

    results_json = app_module._results_export(run.results, run.key, "json")
    assert run.key in results_json
    assert '"simulator"' in results_json, (
        "the exported results must carry the configuration that produced them"
    )


def test_the_page_offers_all_four_files(app):
    """Both artefacts, both formats: the log of what was decided and the scoring of how it
    went, each as a spreadsheet and as structured text."""
    labels = [element.proto.label for element in app.get("download_button")]
    assert labels == [
        "Audit trail (CSV)",
        "Audit trail (JSON)",
        "Results (JSON)",
        "Results (CSV)",
    ]

    urls = [element.proto.url for element in app.get("download_button")]
    assert len(set(urls)) == 4, "two buttons are serving the same file"


def test_a_second_run_replaces_the_first_run_s_files():
    """The bug this section could most plausibly ship with, and the reason the run key is
    in the cache signature.

    The export is cached because Streamlit re-runs the whole script on every interaction
    and a demo-scale trail is thousands of entries. A cache keyed on anything less specific
    than the run would keep handing out the first world's audit trail after the demo moved
    on to a second one — under the *new* run's filename, which is worse than an error,
    because it looks correct.

    Checked by reconstructing the download URL. Streamlit does not put a download's content
    or filename on the element; it stores the bytes and exposes a URL whose id is a hash of
    exactly those three things — content, MIME type and filename. Recomputing that hash
    from the text and name this run *should* have produced therefore verifies all three at
    once, which is as close to opening the downloaded file as a test can get. It reaches
    into a private Streamlit helper to do it; if that helper moves, this test fails loudly
    rather than passing silently, which is the right way round.
    """
    from streamlit.runtime.memory_media_file_storage import _calculate_file_id

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.slider("customers").set_value(60)
    at.sidebar.slider("narration_delay").set_value(0.0)

    seen: list[str] = []
    for seed in (7, 11):
        at.sidebar.number_input("seed").set_value(seed)
        at.sidebar.button[0].click().run()
        assert not at.exception, at.exception

        run = at.session_state["run"]
        assert run.key not in seen, "two different worlds produced the same fingerprint"
        seen.append(run.key)

        # Built from the trail itself, deliberately **not** through the app's cached
        # export helper: an expectation computed by the same cache the test is trying to
        # catch out would go stale in exactly the same way and agree with itself.
        expected_text = run.trail.to_csv()
        expected_name = _load_app_module().export_filename(
            "audit_trail", run.key, run.settings.simulator.seed, "csv"
        )
        expected_id = _calculate_file_id(
            expected_text.encode("utf-8"), "text/csv", expected_name
        )

        url = at.get("download_button")[0].proto.url
        assert url.endswith(f"{expected_id}.csv"), (
            f"the audit trail offered for seed {seed} is not this run's trail under this "
            "run's name"
        )


# ---------------------------------------------------------------------------
# The run controls (C4, operation 5)
#
# Two kinds of property here. One is that a control does what its label says — the
# preset that names a retry storm produces a retry storm, and the storm genuinely makes
# payments fail together, which is the mechanism the pitch has been claiming since
# IDEA.md §8c was written. The other is about the run key: it must change when something
# that affects the outcome changes, and must *not* change when something that cannot
# affect the outcome changes. A fingerprint that moves for no reason is as useless as one
# that stays put when it should move.
# ---------------------------------------------------------------------------


def _fresh_app():
    """An app with the display pause off, ready to have controls set and be run."""
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.sidebar.slider("narration_delay").set_value(0.0)
    return at


def _run(at):
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    return at.session_state["run"]


def test_each_preset_builds_the_world_it_names(app_module):
    """The presets exist so a live demo is not four sliders set while people watch. Each
    must therefore actually land on the world it is named after."""
    at = _fresh_app()

    at.sidebar.selectbox(app_module.PRESET_KEY).set_value(app_module.MASS_FAILURE_PRESET)
    at.sidebar.slider("customers").set_value(60)
    storm = _run(at)
    assert storm.settings.simulator.mass_failure_scenario is True

    at.sidebar.selectbox(app_module.PRESET_KEY).set_value(app_module.STANDARD_PRESET)
    at.sidebar.slider("customers").set_value(60)
    standard = _run(at)
    assert standard.settings.simulator.mass_failure_scenario is False

    assert storm.key != standard.key, (
        "two different worlds were fingerprinted as the same run"
    )


def test_mass_failure_forces_more_simultaneous_failures():
    """The substance behind task 5's headline item: the toggle must reach the simulator
    and change the world, or it is a control that claims a mechanism the demo cannot
    produce.

    It changes two things, and before Phase 13 only the first was true: the *volume* of
    failures the pacing has to spread out, and their *timing* — see the test below.
    """
    at = _fresh_app()
    at.sidebar.slider("customers").set_value(100)

    at.sidebar.toggle("mass_failure").set_value(False)
    calm = _run(at)
    at.sidebar.toggle("mass_failure").set_value(True)
    storm = _run(at)

    def failures(run) -> int:
        return len(run.paired.simulator.failed_billing_event_ids)

    assert failures(storm) > failures(calm) * 1.5, (
        "the mass-failure toggle did not materially increase the number of failures, so "
        "it gives the pacing nothing extra to cope with"
    )
    # A share of the population, not all of it: the point is a surge inside an otherwise
    # normal batch.
    assert failures(storm) < len(storm.paired.simulator.billing_events)


def test_charges_spread_across_the_cycle_until_an_outage_synchronises_them():
    """The fix for the defect a previous version of this test pinned.

    Every customer used to be generated with the same billing cycle *and* the same start
    date, so every charge in a cycle landed on one simulated instant whether the
    mass-failure mode was on or off — which made that mode a volume dial, while
    `SimulatorConfig`, ARCHITECTURE.md A2 operation 10 and IDEA.md 8c all described it as
    creating a synchronisation that was in fact always there.

    Customers now renew on their own anniversary within the cycle
    (`Customer.billing_anniversary_offset_days`), so both halves of that story are real:
    a normal batch is spread out, and the outage genuinely piles a share of it onto one
    instant. That is the condition Jitter exists for.
    """
    from collections import Counter

    at = _fresh_app()
    at.sidebar.slider("customers").set_value(60)

    at.sidebar.toggle("mass_failure").set_value(False)
    calm = _run(at)
    at.sidebar.toggle("mass_failure").set_value(True)
    storm = _run(at)

    def instants(run) -> Counter:
        return Counter(e.scheduled_at for e in run.paired.simulator.billing_events)

    calm_instants = instants(calm)
    assert len(calm_instants) > 1, (
        "a normal batch still lands on a single instant — the anniversary stagger is "
        "not reaching the billing events"
    )
    # Spread across the cycle, not merely across two days.
    assert len(calm_instants) >= 10
    assert max(calm_instants.values()) < len(calm.paired.simulator.billing_events) / 2, (
        "no single day should hold half the batch when renewals are staggered"
    )

    storm_instants = instants(storm)
    biggest_calm_day = max(calm_instants.values())
    biggest_storm_day = max(storm_instants.values())
    assert biggest_storm_day > biggest_calm_day * 2, (
        "the outage did not concentrate charges onto one instant, so the toggle is "
        "still only a volume dial"
    )
    # The forced share lands together: a quarter of the batch on one instant is well
    # beyond anything the stagger produces on its own.
    assert biggest_storm_day >= len(storm.paired.simulator.billing_events) / 4


def test_a_setting_that_cannot_change_the_outcome_does_not_change_the_key():
    """The run key identifies the inputs that determined a run. Two properties follow, and
    this is the easily-missed one: a field that changed nothing must not produce a new
    fingerprint, or a reader comparing two identical runs would be told they differ.

    The mass-failure *share* is inert while the scenario is off, and the narration pause is
    inert always — it moves the screen, never the run, which is exactly what its help text
    promises a judge.
    """
    at = _fresh_app()
    at.sidebar.slider("customers").set_value(60)
    at.sidebar.toggle("mass_failure").set_value(False)

    at.session_state["mass_fraction"] = 0.9
    wide = _run(at)
    at.session_state["mass_fraction"] = 0.1
    narrow = _run(at)
    assert wide.key == narrow.key, (
        "an inert setting changed the fingerprint, so two identical runs would be "
        "reported as different experiments"
    )

    at.sidebar.slider("narration_delay").set_value(0.05)
    paused = _run(at)
    assert paused.key == narrow.key, (
        "the narration pause changed the run key, contradicting what the control says "
        "about itself"
    )


def test_changing_an_agent_setting_is_reported_as_a_different_experiment(app_module):
    """Every figure in this project's notes was recorded under one agent configuration. A
    judge who nudges the explore rate and then reads a recorded number off a slide is
    comparing two experiments — so the page says so at the moment of the nudge, rather
    than leaving it to be discovered later or never.
    """
    at = _fresh_app()
    at.sidebar.slider("customers").set_value(60)
    standard = _run(at)
    assert not at.sidebar.warning, "the standard configuration was flagged as modified"

    at.sidebar.slider("epsilon").set_value(
        app_module.AGENT_DEFAULTS["epsilon"] + 0.25
    )
    changed = _run(at)

    assert changed.key != standard.key
    warnings = [element.value for element in at.sidebar.warning]
    assert warnings, "the agents were changed and the page said nothing"
    assert "different experiment" in warnings[0]
