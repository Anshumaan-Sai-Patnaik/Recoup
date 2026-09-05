# AI Revenue Recovery Agent

A Razorpay AI Buildathon submission (Track 3: Revenue Recovery). An agent that recovers
failed *mandate/subscription* payments (card, UPI Autopay, netbanking standing
instruction) by diagnosing why a payment failed and choosing how/when/whether to retry it
— instead of naively hammering the same channel, which can get a card flagged as fraud and
blocked.

See `notes/IDEA.md` for the full explanation of the problem and approach, from first
principles. See `notes/ARCHITECTURE.md` for the component breakdown, `notes/DESIGN.md` for
the tech stack, and `notes/PLAN.md` + `notes/TRACKER.md` for the phase-wise build plan and
current build status.

## Setup

Python 3.11 or newer, in a fresh virtual environment:

```
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate  on macOS/Linux
pip install -r requirements.txt
```

Six dependencies, no database, no containers, no cloud — `notes/DESIGN.md` explains why
each one is there.

## Run

### The dashboard (live demo)

```
streamlit run dashboard/app.py
```

Pick a preset in the sidebar — a standard world, a bank outage, or a quiet one — then
press **Run batch**. Both agents, the Smart Agent and the naive Baseline, work through the
same seeded set of failed payments, and every decision is narrated in plain English as it
is made.

The preset just sets the controls beneath it, and any of them can be moved afterwards: the
seed (which world), the population size, how often a billing event fails at all, and a
**mass failure** toggle that forces a share of the population onto one simulated instant —
the thundering-herd condition the jitter exists to spread back out. An *Agent
configuration* panel exposes both agents' own settings, and a narration-pause slider sets
how fast the feed reads. The pause is display speed only; the run itself is identical with
or without it.

Below the feed, the head-to-head view scores both agents over that world: recovery rate,
attempts used, escalations, and how many payment methods the agent permanently retired,
plus the outcome mix and a full metric-by-metric table. A second view scores the Smart
Agent against the simulator's private answer key — what it recovered of what was genuinely
recoverable, and why the rest got away.

The audit trail and the results are then downloadable as CSV or JSON, each file named for
the run that produced it. Last on the page, you can pick any single transaction and read
both agents' handling of that one payment side by side.

### Headless (the full pipeline in one call)

Run from the project root, so `recovery_agent` is importable:

```python
from recovery_agent.metrics import run_evaluation
from recovery_agent.simulator import SimulatorConfig

results = run_evaluation(SimulatorConfig(seed=11, num_customers=600, base_failure_rate=0.25))
print(results.label)
print(results.to_frame())
results.to_json("results.json")
```

Builds the world, runs both agents over it, scores them against each other *and* against
the Simulator's private answer key, and returns one exportable object keyed by a
fingerprint of the whole configuration — so any run can be reproduced later from what the
result itself records.

### Tests

```
pytest
```

## Where the numbers come from

Everything runs against a seeded simulator, not real bank rails (`notes/IDEA.md` §11
explains why, and `notes/DESIGN.md` §6 is explicit about what is deliberately *not* real
here). The simulator holds a private "hidden truth" answer key for every failed payment —
whether it was genuinely recoverable, on which channel, and after how long — which the
agents never see and which the evaluation engine uses to mark their work.

Across thirty independent worlds (seeds 11–40, 600 customers each, 4,568 failed billing
events), the Smart Agent recovered **+10.9 percentage points** more than the Baseline
(range +5.7 to +16.1) using **41.2% fewer attempts**, reaching **97.5%** of what was
genuinely recoverable against the Baseline's 82.0%, and escalating a third fewer payments
to a human — and made **zero** retries on a payment method it had itself diagnosed as
permanently dead. Every one of those held on every one of the thirty seeds.

Thirty worlds rather than ten on purpose. An earlier version of these notes quoted seeds
11–20 alone; when the simulator was corrected in Phase 13 and the numbers regenerated,
those ten seeds read +12.7 while twenty unused ones read +10.0. The wider sample is the
honest one. `notes/TRACKER.md` records both.
