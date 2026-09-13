"""Post one week of a CEO-Bench episode to Reef, one report per turn.

The agent (``harness.agent``) routes every model call of an episode through
Reef and knows, from the dashboard each request carries, which simulated week
a call belongs to and the state the week started in. When the next week's
dashboard appears the week is over, and :func:`post_week_reports` turns it
into one Reef report per turn: the week's change in company value over the
starting balance as the score, that turn's receipt as the only reference. One
reference per report is what the ``sao`` recipe trains on.

A week's opening state is valued as its cash plus the subscription run-rate
the dashboard implies (each individual subscriber at the lowest listed price,
each enterprise seat at plan C's), counted over the weeks left in the episode
and at most ``horizon_weeks`` of them (:func:`valuation`). Cash alone made
every purchase a loss and inaction the safest week; the run-rate term is what
pays for growth inside the horizon. The last week of an episode ends with the
verifier's final cash, valued as cash alone, instead of a next dashboard.

A turn longer than the trainer's window (``max_tokens``, prompt and
completion together) is skipped: the engine served it and Reef recorded it,
but the trainer could not hold it, so it stays evaluation-only.
"""

import uuid
from collections.abc import Sequence

from reef_client import ReefClient

#: The benchmark's starting balance (the runner's ``--cash`` default).
INITIAL_CASH = 1_000_000.0
#: Weeks of subscription run-rate a week's opening state is valued at, at most.
DEFAULT_VALUE_HORIZON_WEEKS = 26
#: Subscriptions bill every 30 days; a simulated week is this share of a bill.
DAYS_PER_WEEK, DAYS_PER_BILLING_MONTH = 7, 30


def valuation(cash: float, run_rate: float, remaining_weeks: int, horizon_weeks: int) -> float:
    """Cash plus the monthly ``run_rate`` over the weeks left, at most ``horizon_weeks`` of them."""
    weeks = max(0, min(remaining_weeks, horizon_weeks))
    return cash + run_rate * DAYS_PER_WEEK * weeks / DAYS_PER_BILLING_MONTH


def week_score(value_start: float, value_end: float, initial_cash: float = INITIAL_CASH) -> float:
    """A week's change in company value in units of the starting balance."""
    return (value_end - value_start) / initial_cash


def post_week_reports(
    client: ReefClient,
    scenario: str,
    *,
    week: int,
    day: int,
    cash_start: float,
    cash_end: float,
    value_start: float,
    value_end: float,
    turns: Sequence[tuple[str, int]],
    max_tokens: int = 0,
) -> list[dict]:
    """Report one finished week against each of its turns' receipts.

    ``turns`` are ``(receipt, tokens)`` pairs in call order; the score is the
    week's change in value, the cash figures travel along for the record.
    """
    score = week_score(value_start, value_end)
    feedback = (
        f"ceobench week {week} (from day {day}): value {value_start:.0f} -> {value_end:.0f}"
        f" (cash {cash_start:.0f} -> {cash_end:.0f}), score {score:.4f} over {len(turns)} turns"
    )
    posted = []
    for index, (receipt, tokens) in enumerate(turns):
        if max_tokens and tokens > max_tokens:
            continue
        payload = {
            # A report id derived from the receipt makes a duplicate post a
            # no-op on Reef's side, not a second report about the same turn.
            "agent_record_id": uuid.uuid5(uuid.NAMESPACE_URL, f"reef:ceobench:{receipt}").hex,
            "score": score,
            "feedback": feedback,
            "references": [receipt],
            "metadata": {
                "ceobench": {
                    "week": week,
                    "day": day,
                    "cash_start": cash_start,
                    "cash_end": cash_end,
                    "value_start": value_start,
                    "value_end": value_end,
                    "turn": index,
                    "turns": len(turns),
                }
            },
        }
        posted.append(client.report(scenario, payload))
    return posted
