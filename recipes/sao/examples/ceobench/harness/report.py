"""Post one week of a CEO-Bench episode to Reef, one report per turn.

The agent (``harness.agent``) routes every model call of an episode through
Reef and knows, from the dashboard each request carries, which simulated week
a call belongs to and the state the week started in. When enough later weeks
have opened the week is scored, and :func:`post_week_reports` turns it into
one Reef report per turn: the week's credit as the score, that turn's receipt
as the only reference. One reference per report is what the ``sao`` recipe
trains on.

A week's opening state is valued as its cash plus the subscription run-rate
the dashboard implies (each individual subscriber at the lowest listed price,
each enterprise seat at plan C's), counted over the weeks left in the episode
and at most ``horizon_weeks`` of them (:func:`valuation`). Cash alone made
every purchase a loss and inaction the safest week; the run-rate term is what
pays for growth inside the horizon.

A week's credit is the discounted sum of the value changes of the next
``credit_weeks`` weeks (:func:`week_credit`): the week that spends on
acquisition is credited with the subscribers that arrive over the weeks
after it. The last weeks of an episode end with the verifier's final cash,
valued as cash alone, and their sums are cut short there.

Credits are posted through :class:`ScoreScale`, which clips one week's
outliers (the benchmark's six-figure R&D purchases) and divides by the
running median magnitude, so an ordinary week's difference from the one
before it keeps a gradient instead of vanishing next to the outliers.

A turn longer than the trainer's window (``max_tokens``, prompt and
completion together) is skipped: the engine served it and Reef recorded it,
but the trainer could not hold it, so it stays evaluation-only.
"""

import statistics
import uuid
from collections.abc import Sequence

from reef_client import ReefClient

#: The benchmark's starting balance (the runner's ``--cash`` default).
INITIAL_CASH = 1_000_000.0
#: Weeks of subscription run-rate a week's opening state is valued at, at most.
DEFAULT_VALUE_HORIZON_WEEKS = 26
#: Subscriptions bill every 30 days; a simulated week is this share of a bill.
DAYS_PER_WEEK, DAYS_PER_BILLING_MONTH = 7, 30
#: Weeks of value change a week is credited with, and the discount per week.
DEFAULT_CREDIT_WEEKS = 4
DEFAULT_CREDIT_DISCOUNT = 0.8
#: A week's credit is clipped to this magnitude (in units of the starting
#: balance) before scaling, and the scale never drops below the floor.
DEFAULT_SCORE_CLIP = 0.05
DEFAULT_SCORE_FLOOR = 0.003
#: Scaled scores stay within this magnitude.
SCORE_CAP = 3.0


def valuation(cash: float, run_rate: float, remaining_weeks: int, horizon_weeks: int) -> float:
    """Cash plus the monthly ``run_rate`` over the weeks left, at most ``horizon_weeks`` of them."""
    weeks = max(0, min(remaining_weeks, horizon_weeks))
    return cash + run_rate * DAYS_PER_WEEK * weeks / DAYS_PER_BILLING_MONTH


def week_credit(deltas: Sequence[float], discount: float, initial_cash: float = INITIAL_CASH) -> float:
    """The discounted sum of the value changes in ``deltas``, in units of the starting balance.

    ``deltas[0]`` is the week's own change, ``deltas[1]`` the next week's, and
    so on; a shorter sequence is an episode that ended inside the window.
    """
    return sum(delta * discount**position for position, delta in enumerate(deltas)) / initial_cash


class ScoreScale:
    """Scale each week's credit by the running median magnitude of the credits so far.

    The credit is clipped to ``clip`` first, so one six-figure purchase does
    not set the scale for the rest of the episode; the scale never drops
    below ``floor``, so a run of near-zero weeks does not blow small noise up
    to the cap. Scaling keeps the sign: a week that grew the company scores
    positive whatever the weeks around it did.
    """

    def __init__(self, clip: float = DEFAULT_SCORE_CLIP, floor: float = DEFAULT_SCORE_FLOOR) -> None:
        if clip <= 0 or floor <= 0:
            raise ValueError("score clip and floor must be positive")
        self.clip = float(clip)
        self.floor = float(floor)
        self.magnitudes: list[float] = []

    def scale(self, credit: float) -> float:
        clipped = max(-self.clip, min(self.clip, credit))
        self.magnitudes.append(abs(clipped))
        scale = max(statistics.median(self.magnitudes), self.floor)
        return max(-SCORE_CAP, min(SCORE_CAP, clipped / scale))


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
    credit: float,
    score: float,
    turns: Sequence[tuple[str, int]],
    max_tokens: int = 0,
) -> list[dict]:
    """Report one finished week against each of its turns' receipts.

    ``turns`` are ``(receipt, tokens)`` pairs in call order; ``score`` is the
    scaled ``credit`` and is what Reef trains on, the rest travels along for
    the record.
    """
    feedback = (
        f"ceobench week {week} (from day {day}): credit {credit:.4f}, score {score:.2f};"
        f" value {value_start:.0f} -> {value_end:.0f} (cash {cash_start:.0f} -> {cash_end:.0f})"
        f" over {len(turns)} turns"
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
                    "credit": credit,
                    "turn": index,
                    "turns": len(turns),
                }
            },
        }
        posted.append(client.report(scenario, payload))
    return posted
