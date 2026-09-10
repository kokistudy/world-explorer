"""
Voting network model data structures, plus pivotality/responsibility/
criticality/contingency analysis.

Terminology:
  - threshold:  probability of voting Y on a given day (higher = more likely).
                Starts at base_threshold; only rises when a Y-voting neighbor
                exerts influence. Non-voting neighbors have no effect.
  - weight:     directed influence strength from one agent to another [0, 1]
  - pivot:      an agent whose vote(s) were causally decisive for the outcome
  - responsibility: graded version of pivotality -- fraction of noisy
                counterfactual re-runs (that agent absent) in which the
                proposal fails. Ex-post: conditions on the actual run.
                The default locks others' draws to the "they signed / didn't"
                range. pivotality_unlocked_later instead freezes only earlier
                signers and uses fresh Unif(0,1) draws from the target's day on.
  - criticality: ex-ante credit/blame/responsibility (Gerstenberg, Lagnado, &
                Zultan, 2023) -- how important an agent's contribution is
                expected to be, computed from the network structure and base
                thresholds alone, before any signing has happened.
  - contingency: probability-contingency / crediting causality (Spellman,
                1997) -- the change in the proposal's probability of
                eventually passing, from right before a given agent signs to
                right after, crediting only that agent's own sign.

Run as a script to compute predicted pivotality/responsibility/criticality/
contingency for an existing trial (config.json + draws.json + day_signed.json
already on disk):

Usage:
    python run_model.py 1
    python run_model.py 3 --pilot
    python run_model.py 2 --n_samples 2000 --seed 42
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Static network description (shared across all runs / Monte Carlo samples)
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """One participant in the voting network."""
    name: str                        # e.g. "Alice", "A"
    base_threshold: float            # starting propensity to vote Y, in [0, 1]

    def to_dict(self) -> dict:
        return {"name": self.name, "base_threshold": self.base_threshold}

    @staticmethod
    def from_dict(d: dict) -> Agent:
        return Agent(name=d["name"], base_threshold=d["base_threshold"])


@dataclass
class NetworkSpec:
    """
    The fixed topology and parameters of the voting network.

    Influence weights are stored as a dict of dicts:
        influence[i][j] = w  means agent i influences agent j with strength w.
    So if Alice -> Bob with w=0.3, then influence["Alice"]["Bob"] = 0.3.

    Threshold update rule:
        If neighbor j voted Y on day t, then on day t+1:
            thresh[i] = min(1, thresh[i] + w(j→i))
        Non-voting neighbors have no effect.
    """
    agents: list[Agent]
    influence: dict[str, dict[str, float]]   # influence[src][dst] = weight
    n_days: int                              # total days D
    pass_threshold: int = 1                  # number of Y votes needed to pass

    @property
    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents]

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def to_dict(self) -> dict:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "influence": self.influence,
            "n_days": self.n_days,
            "pass_threshold": self.pass_threshold,
        }

    @staticmethod
    def from_dict(d: dict) -> NetworkSpec:
        return NetworkSpec(
            agents=[Agent.from_dict(a) for a in d["agents"]],
            influence=d["influence"],
            n_days=d["n_days"],
            pass_threshold=d["pass_threshold"],
        )


# ---------------------------------------------------------------------------
# Per-day snapshot (one entry in the simulation log)
# ---------------------------------------------------------------------------

@dataclass
class DayState:
    """Everything that happened on a single day of the simulation."""
    day: int                                    # 1-indexed

    # Threshold each agent entered this day with (before any votes)
    thresholds_start: dict[str, float]

    # Random draw for each agent who had not yet voted (None if already voted)
    random_draws: dict[str, Optional[float]]

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "thresholds_start": self.thresholds_start,
            "random_draws": self.random_draws,
        }

    @staticmethod
    def from_dict(d: dict) -> DayState:
        return DayState(
            day=d["day"],
            thresholds_start=d["thresholds_start"],
            random_draws=d["random_draws"],
        )


# ---------------------------------------------------------------------------
# Full record of one simulation run
# ---------------------------------------------------------------------------

@dataclass
class SimulationRun:
    """
    Complete record of one run of the voting simulation.

    This is the main artifact you'll mine for causality/pivot analysis.
    """
    spec: NetworkSpec
    seed: Optional[int]                  # RNG seed for reproducibility

    # One entry per day, in order
    days: list[DayState] = field(default_factory=list)

    # day_voted[agent_name] = day on which they cast their Y vote, or None
    day_voted: dict[str, Optional[int]] = field(default_factory=dict)

    # Causality scores (filled in by analysis code, not the simulator)
    pivot_scores: dict[str, float] = field(default_factory=dict)

    # --- Derived properties ---

    @property
    def yes_voters(self) -> list[str]:
        return [name for name, day in self.day_voted.items() if day is not None]

    @property
    def no_voters(self) -> list[str]:
        return [name for name, day in self.day_voted.items() if day is None]

    @property
    def final_yes_count(self) -> int:
        return len(self.yes_voters)

    @property
    def passed(self) -> bool:
        return self.final_yes_count >= self.spec.pass_threshold

    @property
    def passage_day(self) -> Optional[int]:
        required = self.spec.pass_threshold
        yes_days = sorted(d for d in self.day_voted.values() if d is not None)
        for i, day in enumerate(yes_days, 1):
            if i >= required:
                return day
        return None

    # --- JSON serialization ---

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "seed": self.seed,
            "day_voted": self.day_voted,
            "days": [d.to_dict() for d in self.days],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def save_json(self, path: str, **kwargs) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, **kwargs)

    @staticmethod
    def from_dict(d: dict) -> SimulationRun:
        return SimulationRun(
            spec=NetworkSpec.from_dict(d["spec"]),
            seed=d["seed"],
            day_voted=d["day_voted"],
            days=[DayState.from_dict(ds) for ds in d["days"]],
            pivot_scores=d.get("pivot_scores", {}),
        )

    @staticmethod
    def from_json(s: str) -> SimulationRun:
        return SimulationRun.from_dict(json.loads(s))

    @staticmethod
    def load_json(path: str) -> SimulationRun:
        with open(path) as f:
            return SimulationRun.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Threshold update helper
# ---------------------------------------------------------------------------

def update_thresholds(
    current: dict[str, float],
    voted_yes_today: set[str],
    spec: NetworkSpec,
) -> dict[str, float]:
    """
    Compute next-day thresholds after one round of voting.

    For each agent j that voted Y today, additively increase each downstream
    neighbor i's threshold by the connection strength:
        thresh[i] = min(1, thresh[i] + w(j→i))

    Agents who did not vote Y have no effect on their neighbors.
    """
    new_thresh = dict(current)
    for src in voted_yes_today:
        for dst, weight in spec.influence.get(src, {}).items():
            new_thresh[dst] = min(1.0, new_thresh[dst] + weight)
    return new_thresh


# ---------------------------------------------------------------------------
# Pivotality analysis
# ---------------------------------------------------------------------------

def _remove_agent(spec: NetworkSpec, excluded: str) -> NetworkSpec:
    """A copy of `spec` with `excluded` removed entirely: no vote, no influence."""
    restricted_influence: dict[str, dict[str, float]] = {
        src: {dst: w for dst, w in dsts.items() if dst != excluded}
        for src, dsts in spec.influence.items()
        if src != excluded
    }
    return NetworkSpec(
        agents=[a for a in spec.agents if a.name != excluded],
        influence=restricted_influence,
        n_days=spec.n_days,
        pass_threshold=spec.pass_threshold,
    )


def _resimulate_without(run: SimulationRun, excluded: str) -> bool:
    """Re-run with `excluded` absent (no votes, no influence), same random draws.
    Returns True if the proposal still passes."""
    spec = run.spec
    draws_by_day = {ds.day: ds.random_draws for ds in run.days}
    restricted_spec = _remove_agent(spec, excluded)

    thresholds = {a.name: a.base_threshold for a in restricted_spec.agents}
    voted_yes: set[str] = set()

    for day in range(1, spec.n_days + 1):
        draws = draws_by_day[day]
        voted_yes_today = {
            name for name in thresholds.keys() - voted_yes
            if draws[name] < thresholds[name]
        }
        voted_yes |= voted_yes_today
        thresholds = update_thresholds(thresholds, voted_yes_today, restricted_spec)

    return len(voted_yes) >= spec.pass_threshold


def compute_pivotality(run: SimulationRun) -> dict[str, bool]:
    """Return pivot status for each yes-voter: True if their absence would have
    caused the proposal to fail, holding all random draws fixed."""
    if not run.passed:
        return {name: False for name in run.yes_voters}
    return {name: not _resimulate_without(run, name) for name in run.yes_voters}


def _resimulate_without_noisy(
    run: SimulationRun,
    excluded: str,
    rng: "numpy.random.Generator",
) -> bool:
    """
    One noisy counterfactual run with `excluded` absent.

    Draws for remaining agents are resampled from ranges consistent with the
    original run:
      - Agent voted on day D  → draw was in [0, original_threshold_D)
      - Agent didn't vote day D → draw was in [original_threshold_D, 1]

    The sampled draw is then compared against the *counterfactual* threshold
    (which may differ because the excluded agent's influence is removed).
    """
    spec = run.spec
    orig_thresh_by_day = {ds.day: ds.thresholds_start for ds in run.days}
    orig_day_voted = run.day_voted

    restricted_influence: dict[str, dict[str, float]] = {
        src: {dst: w for dst, w in dsts.items() if dst != excluded}
        for src, dsts in spec.influence.items()
        if src != excluded
    }
    restricted_spec = NetworkSpec(
        agents=[a for a in spec.agents if a.name != excluded],
        influence=restricted_influence,
        n_days=spec.n_days,
        pass_threshold=spec.pass_threshold,
    )

    thresholds = {a.name: a.base_threshold for a in restricted_spec.agents}
    voted_yes: set[str] = set()

    for day in range(1, spec.n_days + 1):
        orig_thresh_today = orig_thresh_by_day[day]
        voted_yes_today: set[str] = set()

        for name in thresholds.keys() - voted_yes:
            orig_thresh = orig_thresh_today[name]
            voted_day = orig_day_voted.get(name)

            if voted_day == day:
                draw = rng.uniform(0, orig_thresh)
            else:
                draw = rng.uniform(orig_thresh, 1)

            if draw < thresholds[name]:
                voted_yes_today.add(name)

        voted_yes |= voted_yes_today
        thresholds = update_thresholds(thresholds, voted_yes_today, restricted_spec)

    return len(voted_yes) >= spec.pass_threshold


def compute_responsibility(
    run: SimulationRun,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> dict[str, float]:
    """
    Graded responsibility for each agent: the fraction of noisy counterfactual
    simulations (with that agent absent) in which the proposal fails.

    Draws are resampled from ranges consistent with the original run's
    observed thresholds, but compared against counterfactual thresholds.
    Non-voters get 0.0.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    all_agents = [a.name for a in run.spec.agents]
    if not run.passed:
        return {name: 0.0 for name in all_agents}

    result = {}
    for name in all_agents:
        if name not in run.yes_voters:
            result[name] = 0.0
            continue
        fail_count = sum(
            not _resimulate_without_noisy(run, name, rng)
            for _ in range(n_samples)
        )
        result[name] = fail_count / n_samples
    return result


def compute_responsibility_unlocked_later(
    run: SimulationRun,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> dict[str, float]:
    """
    Graded pivotality that does not lock same-day / later draws to the
    observed "they signed" range.

    History through day t-1 is frozen (who already signed, and thresholds at
    the start of day t). The target is removed from that day on. Everyone
    still unsigned -- including people who actually signed on day t or later
    -- gets a fresh Unif(0,1) draw each remaining day, compared to their
    counterfactual threshold. Non-signers and failed trials get 0.0.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    spec = run.spec
    all_agents = spec.agent_names
    if not run.passed:
        return {name: 0.0 for name in all_agents}

    thresh_by_day = {ds.day: ds.thresholds_start for ds in run.days}
    orig_day_voted = run.day_voted
    result = {}
    for name in all_agents:
        t = orig_day_voted.get(name)
        if t is None:
            result[name] = 0.0
            continue
        signed_already = {
            n for n, d in orig_day_voted.items()
            if d is not None and d < t
        }
        thresholds = dict(thresh_by_day[t])
        fail_count = 0
        for _ in range(n_samples):
            draws = _draw_tail(spec, t, rng)
            n_signed = _simulate_tail_with_draws(
                spec, t, thresholds, signed_already, draws,
                excluded={name},
            )
            if n_signed < spec.pass_threshold:
                fail_count += 1
        result[name] = fail_count / n_samples
    return result


# ---------------------------------------------------------------------------
# Criticality analysis (ex-ante, before any signing has happened)
#
# Extends the anticipated-credit / anticipated-blame / anticipated-responsibility
# models of Gerstenberg, Lagnado, & Zultan (2023) from their static, independent-
# actions setting to a network with influence and a day-by-day time dimension.
# In their setting, "player i succeeds" or "player i fails" is a single
# exogenous coin flip. Here, whether an agent signs is itself the outcome of a
# multi-day process shaped by the network's influence, so a counterfactual on
# "agent i's action" has to pick a concrete alternative history:
#   - do(o_i = fail):    agent i is removed from the network for the whole
#                         run -- never signs, exerts no influence on anyone.
#   - do(o_i = succeed): agent i is forced to sign on day 1 -- the earliest,
#                         most-influential way for them to succeed -- while
#                         remaining in the network and influencing others as
#                         usual from day 1 onward.
# Both counterfactuals share the same underlying random draws as the baseline
# sample (common random numbers), so the only thing that differs is agent i's
# status.
# ---------------------------------------------------------------------------

def _simulate_with_draws(
    spec: NetworkSpec,
    draws: dict[int, dict[str, float]],
    forced_yes_day1: Optional[set[str]] = None,
) -> dict[str, Optional[int]]:
    """
    Run the day-by-day signing process forward given pre-drawn random numbers
    for every (day, agent) pair. Agents in `forced_yes_day1` sign on day 1
    regardless of their draw. Returns {agent: day_signed_or_None}.
    """
    forced_yes_day1 = forced_yes_day1 or set()
    thresholds = {a.name: a.base_threshold for a in spec.agents}
    voted_yes: set[str] = set()
    day_voted: dict[str, Optional[int]] = {}

    for day in range(1, spec.n_days + 1):
        voted_yes_today = {
            name for name in thresholds.keys() - voted_yes
            if (name in forced_yes_day1 and day == 1) or draws[day][name] < thresholds[name]
        }
        for name in voted_yes_today:
            day_voted[name] = day
        voted_yes |= voted_yes_today
        thresholds = update_thresholds(thresholds, voted_yes_today, spec)

    for name in thresholds.keys() - voted_yes:
        day_voted[name] = None
    return day_voted


def compute_criticality(
    spec: NetworkSpec,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> dict[str, dict[str, float]]:
    """
    Monte Carlo estimate of each agent's ex-ante criticality, before any
    signing has happened, based only on the network's structure and base
    thresholds.

    Returns {agent: {"credit": ..., "blame": ..., "responsibility": ...}}:
      - credit:         Pr(do(o_i=fail) fails | proposal actually passes)
      - blame:          Pr(do(o_i=succeed) passes | proposal actually fails)
      - responsibility: credit * Pr(passes) + blame * Pr(fails)
    (Eqs. 3-5 in Gerstenberg, Lagnado, & Zultan, 2023.)
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    names = spec.agent_names
    restricted_specs = {name: _remove_agent(spec, name) for name in names}

    win_total = 0
    loss_total = 0
    credit_hits = {name: 0 for name in names}
    blame_hits = {name: 0 for name in names}

    for _ in range(n_samples):
        draws = {
            day: {name: rng.uniform() for name in names}
            for day in range(1, spec.n_days + 1)
        }
        baseline_voted = _simulate_with_draws(spec, draws)
        baseline_pass = sum(d is not None for d in baseline_voted.values()) >= spec.pass_threshold

        if baseline_pass:
            win_total += 1
        else:
            loss_total += 1

        for name in names:
            if baseline_pass:
                cf_voted = _simulate_with_draws(restricted_specs[name], draws)
                cf_pass = sum(d is not None for d in cf_voted.values()) >= spec.pass_threshold
                if not cf_pass:
                    credit_hits[name] += 1
            else:
                cf_voted = _simulate_with_draws(spec, draws, forced_yes_day1={name})
                cf_pass = sum(d is not None for d in cf_voted.values()) >= spec.pass_threshold
                if cf_pass:
                    blame_hits[name] += 1

    p_win = win_total / n_samples
    p_loss = loss_total / n_samples

    result = {}
    for name in names:
        credit = credit_hits[name] / win_total if win_total else 0.0
        blame = blame_hits[name] / loss_total if loss_total else 0.0
        result[name] = {
            "credit": credit,
            "blame": blame,
            "responsibility": credit * p_win + blame * p_loss,
        }
    return result


# ---------------------------------------------------------------------------
# Contingency analysis (Spellman, 1997) -- probability-of-success before vs.
# after a given agent's own sign, holding everyone else's actual history fixed
# up to that point and simulating the rest of the run forward stochastically.
# ---------------------------------------------------------------------------

def _simulate_tail_with_draws(
    spec: NetworkSpec,
    start_day: int,
    thresholds: dict[str, float],
    signed: set,
    draws: dict[int, dict[str, float]],
    forced_today: Optional[set] = None,
    excluded: Optional[set] = None,
    blocked_today: Optional[set] = None,
) -> int:
    """Finish the run from `start_day` onward given pre-drawn random numbers
    for every (day, agent) pair from `start_day` on, plus who has already
    signed and everyone's current threshold. Agents in `forced_today` are
    guaranteed to sign on `start_day` itself; agents in `blocked_today` cannot
    sign on `start_day` but may still sign later; everyone else (except those
    in `excluded`) signs whenever their draw falls below their threshold.
    Agents in `excluded` never sign for the rest of the run. Returns the
    final signed count."""
    forced_today = forced_today or set()
    excluded = excluded or set()
    blocked_today = blocked_today or set()
    thresholds = dict(thresholds)
    signed = set(signed)
    for day in range(start_day, spec.n_days + 1):
        voted_today = {
            name for name in spec.agent_names
            if name not in signed
            and name not in excluded
            and not (day == start_day and name in blocked_today)
            and (
                (day == start_day and name in forced_today)
                or draws[day][name] < thresholds[name]
            )
        }
        signed |= voted_today
        thresholds = update_thresholds(thresholds, voted_today, spec)
    return len(signed)


def _draw_tail(spec: NetworkSpec, start_day: int, rng) -> dict[int, dict[str, float]]:
    """One table of random numbers for every (day, agent) from `start_day` on."""
    return {
        d: {agent: rng.uniform() for agent in spec.agent_names}
        for d in range(start_day, spec.n_days + 1)
    }


def compute_contingency(
    run: SimulationRun,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> dict[str, float]:
    """
    Probability-contingency / crediting-causality (Spellman, 1997) for each
    agent who signed: the change in the proposal's probability of eventually
    passing, from right before that agent signs to right after -- isolating
    only that agent's own sign, while still letting everyone else have their
    normal stochastic chance to sign that same day in both cases.

    "Before": everyone's real history up through the prior day is held
    fixed; from this agent's signing day onward, everyone (including this
    agent) signs whenever their random draw falls below their threshold.
    "After": identical, except this agent is additionally guaranteed to sign
    on that same day -- everyone else still signs stochastically that day as
    usual, so the only thing isolated is this agent's own action.

    Before and after share the same random draws per Monte Carlo sample
    (common random numbers), so within a sample "after" can only sign a
    superset of who "before" signed -- the estimated contingency is
    therefore non-negative for every sample, not just in expectation.

    Agents who sign once the pass threshold has already been reached by
    earlier signers get contingency 0.0 (sanity check). Non-signers get 0.0.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    spec = run.spec

    result = {name: 0.0 for name in spec.agent_names}
    for name in run.yes_voters:
        day = run.day_voted[name]
        signed_before = {n for n, d in run.day_voted.items() if d is not None and d < day}

        if len(signed_before) >= spec.pass_threshold:
            result[name] = 0.0
            continue

        thresholds_start = run.days[day - 1].thresholds_start

        before_hits = 0
        after_hits = 0
        for _ in range(n_samples):
            draws = {
                d: {agent: rng.uniform() for agent in spec.agent_names}
                for d in range(day, spec.n_days + 1)
            }
            before_pass = _simulate_tail_with_draws(
                spec, day, thresholds_start, signed_before, draws
            ) >= spec.pass_threshold
            after_pass = _simulate_tail_with_draws(
                spec, day, thresholds_start, signed_before, draws, forced_today={name}
            ) >= spec.pass_threshold
            before_hits += before_pass
            after_hits += after_pass

        result[name] = after_hits / n_samples - before_hits / n_samples

    return result


# ---------------------------------------------------------------------------
# Split contingency -- each signer as last on their day: Y - Z.
#   Y = P(pass) after all of that day's actual signers are forced
#   Z = P(pass) with everyone else that day forced, this person not
# Two rules for the person who is "not in" at Z:
#   can_sign_later: blocked from signing today; may still sign later
#   never_signs: they never sign for the rest of the trial
# Y and Z share the same random draws (pairing).
# ---------------------------------------------------------------------------

def compute_contingency_split(
    run: SimulationRun,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    never_signs: bool = False,
) -> dict[str, float]:
    """
    Same-day contingency, each person as last.

    For a signer on day t, hold the other people who signed that day as
    already in, and take P(pass | all of them in) minus P(pass | the others
    in, this person not). If never_signs is False, "not" means blocked from
    signing today but allowed later; if True, they never sign again.
    Shares need not add to the day's Y - X.
    """
    import numpy as np
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    spec = run.spec
    result = {name: 0.0 for name in spec.agent_names}

    signers_by_day: dict[int, list[str]] = defaultdict(list)
    for name in run.yes_voters:
        signers_by_day[run.day_voted[name]].append(name)

    for day in sorted(signers_by_day):
        day_signers = signers_by_day[day]
        signed_before = {
            n for n, d in run.day_voted.items() if d is not None and d < day
        }
        if len(signed_before) >= spec.pass_threshold:
            continue

        thresholds_start = run.days[day - 1].thresholds_start
        day_signer_set = set(day_signers)
        samples = [_draw_tail(spec, day, rng) for _ in range(n_samples)]

        y_hits = 0
        z_hits = {name: 0 for name in day_signers}
        for draws in samples:
            y_hits += _simulate_tail_with_draws(
                spec, day, thresholds_start, signed_before, draws,
                forced_today=day_signer_set,
            ) >= spec.pass_threshold
            for name in day_signers:
                others = day_signer_set - {name}
                z_hits[name] += _simulate_tail_with_draws(
                    spec, day, thresholds_start, signed_before, draws,
                    forced_today=others,
                    excluded={name} if never_signs else set(),
                    blocked_today=set() if never_signs else {name},
                ) >= spec.pass_threshold

        y = y_hits / n_samples
        for name in day_signers:
            result[name] = y - z_hits[name] / n_samples

    return result


def compute_contingency_simple(
    run: SimulationRun,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> dict[str, float]:
    """
    Equal-split contingency: on each signing day, every co-signer gets
    (Y - X) / n. X is P(pass) at the start of the day; Y is P(pass) after
    all of that day's actual signers are forced. Pairing: same draws for X
    and Y.
    """
    import numpy as np
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    spec = run.spec
    result = {name: 0.0 for name in spec.agent_names}

    signers_by_day: dict[int, list[str]] = defaultdict(list)
    for name in run.yes_voters:
        signers_by_day[run.day_voted[name]].append(name)

    for day in sorted(signers_by_day):
        day_signers = signers_by_day[day]
        signed_before = {
            n for n, d in run.day_voted.items() if d is not None and d < day
        }
        if len(signed_before) >= spec.pass_threshold:
            continue

        thresholds_start = run.days[day - 1].thresholds_start
        day_signer_set = set(day_signers)
        x_hits = 0
        y_hits = 0
        for _ in range(n_samples):
            draws = _draw_tail(spec, day, rng)
            x_hits += _simulate_tail_with_draws(
                spec, day, thresholds_start, signed_before, draws,
            ) >= spec.pass_threshold
            y_hits += _simulate_tail_with_draws(
                spec, day, thresholds_start, signed_before, draws,
                forced_today=day_signer_set,
            ) >= spec.pass_threshold
        share = (y_hits - x_hits) / n_samples / len(day_signers)
        for name in day_signers:
            result[name] = share

    return result


def compute_contingency_shapley(
    run: SimulationRun,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    never_signs: bool = False,
) -> dict[str, float]:
    """
    Same-day Shapley contingency: average sequential increment over every
    order of that day's actual signers. Shares sum to Y - X.

    Pairing: every coalition on a day is replayed on the same random draws.
    never_signs=False is the default: uncredited co-signers are blocked
    today but may still sign later.
    """
    import numpy as np
    from collections import defaultdict
    from itertools import combinations
    from math import factorial

    rng = np.random.default_rng(seed)
    spec = run.spec
    result = {name: 0.0 for name in spec.agent_names}

    signers_by_day: dict[int, list[str]] = defaultdict(list)
    for name in run.yes_voters:
        signers_by_day[run.day_voted[name]].append(name)

    for day in sorted(signers_by_day):
        day_signers = list(signers_by_day[day])
        signed_before = {
            n for n, d in run.day_voted.items() if d is not None and d < day
        }
        if len(signed_before) >= spec.pass_threshold:
            continue

        thresholds_start = run.days[day - 1].thresholds_start
        n = len(day_signers)
        subsets = []
        for r in range(n + 1):
            subsets.extend(combinations(day_signers, r))

        hits = {frozenset(s): 0 for s in subsets}
        samples = [_draw_tail(spec, day, rng) for _ in range(n_samples)]
        for draws in samples:
            for s in subsets:
                sset = frozenset(s)
                not_in = set(day_signers) - sset
                hits[sset] += _simulate_tail_with_draws(
                    spec, day, thresholds_start, signed_before, draws,
                    forced_today=set(sset),
                    excluded=not_in if never_signs else set(),
                    blocked_today=set() if never_signs else not_in,
                ) >= spec.pass_threshold

        n_fact = factorial(n)
        for name in day_signers:
            others = [x for x in day_signers if x != name]
            phi = 0.0
            for r in range(n):
                weight = factorial(r) * factorial(n - r - 1) / n_fact
                for s in combinations(others, r):
                    sset = frozenset(s)
                    phi += weight * (
                        hits[sset | {name}] - hits[sset]
                    ) / n_samples
            result[name] = phi

    return result


# ---------------------------------------------------------------------------
# CLI: load an existing trial from disk and report pivotality/responsibility
# ---------------------------------------------------------------------------

def load_run(trial_dir: "Path") -> SimulationRun:
    with open(trial_dir / "config.json") as f:
        spec = NetworkSpec.from_dict(json.load(f))
    with open(trial_dir / "draws.json") as f:
        output = json.load(f)
    with open(trial_dir / "day_signed.json") as f:
        day_voted = json.load(f)
    return SimulationRun(
        spec=spec,
        seed=output.get("seed"),
        days=[DayState.from_dict(d) for d in output["days"]],
        day_voted=day_voted,
    )


def _process_trial(trial_name: str, trial_dir: "Path", args) -> list[list]:
    """Print one trial's report and return its CSV rows."""
    run = load_run(trial_dir)

    print(f"Trial '{trial_name}' ({trial_dir}/)")
    print(f"  passed: {run.passed}  ({run.final_yes_count}/{run.spec.n_agents} signed, "
          f"needed {run.spec.pass_threshold})")

    responsibility = compute_responsibility(run, n_samples=args.n_samples, seed=args.seed)
    responsibility_unlocked = compute_responsibility_unlocked_later(
        run, n_samples=args.n_samples, seed=args.seed
    )
    criticality = compute_criticality(
        run.spec, n_samples=args.criticality_samples, seed=args.criticality_seed
    )
    contingency = compute_contingency(
        run, n_samples=args.contingency_samples, seed=args.contingency_seed
    )
    split_seed = args.contingency_split_seed
    if split_seed is None:
        split_seed = args.contingency_seed
    split_later = compute_contingency_split(
        run, n_samples=args.contingency_samples, seed=split_seed, never_signs=False
    )
    split_never = compute_contingency_split(
        run, n_samples=args.contingency_samples, seed=split_seed, never_signs=True
    )
    split_simple = compute_contingency_simple(
        run, n_samples=args.contingency_samples, seed=split_seed
    )
    shapley_later = compute_contingency_shapley(
        run, n_samples=args.contingency_samples, seed=split_seed, never_signs=False
    )
    shapley_never = compute_contingency_shapley(
        run, n_samples=args.contingency_samples, seed=split_seed, never_signs=True
    )

    print(f"\n{'agent':<12}{'signed day':<12}{'pivotality':<12}{'piv_unlock':<12}"
          f"{'ant_credit':<12}{'ant_blame':<11}"
          f"{'ant_resp.':<11}{'cont_old':<10}{'can_later':<12}{'split_never':<12}"
          f"{'simple':<10}{'shap_later':<12}{'shap_never':<12}")
    rows = []
    for name in run.spec.agent_names:
        day = run.day_voted.get(name)
        resp = responsibility.get(name, 0.0)
        resp_unlock = responsibility_unlocked.get(name, 0.0)
        crit = criticality[name]
        cont = contingency.get(name, 0.0)
        later = split_later.get(name, 0.0)
        never = split_never.get(name, 0.0)
        simple = split_simple.get(name, 0.0)
        shap_later = shapley_later.get(name, 0.0)
        shap_never = shapley_never.get(name, 0.0)
        print(f"{name:<12}{str(day):<12}{resp:<12.3f}{resp_unlock:<12.3f}"
              f"{crit['credit']:<12.3f}"
              f"{crit['blame']:<11.3f}{crit['responsibility']:<11.3f}{cont:<10.3f}"
              f"{later:<12.3f}{never:<12.3f}{simple:<10.3f}"
              f"{shap_later:<12.3f}{shap_never:<12.3f}")
        rows.append([
            trial_name, name, day, resp, resp_unlock, crit["credit"], crit["blame"],
            crit["responsibility"], cont, later, never, simple,
            shap_later, shap_never,
        ])
    print()

    return rows


def main():
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Compute predicted pivotality/responsibility/criticality for an existing "
                     "trial, or for every trial in --trials_dir if none is given."
    )
    parser.add_argument("trial", nargs="?", default=None,
                         help="Trial name or number (subdirectory of --trials_dir). "
                              "If omitted, runs on every trial subdirectory found.")
    parser.add_argument("--trials_dir", type=str, default=None,
                         help="Directory containing trial subfolders. Defaults to \"trials\", "
                              "or \"pilot_trials\" if --pilot is given.")
    parser.add_argument("--pilot", action="store_true",
                         help="Look up the given trial number under pilot_trials/ instead of trials/ "
                              "(shorthand for --trials_dir pilot_trials).")
    parser.add_argument("--n_samples", type=int, default=1000,
                         help="Number of noisy counterfactual re-runs per agent for responsibility.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for responsibility's noisy resampling.")
    parser.add_argument("--criticality_samples", type=int, default=1000,
                         help="Number of Monte Carlo forward simulations for ex-ante criticality.")
    parser.add_argument("--criticality_seed", type=int, default=None,
                         help="Seed for criticality's Monte Carlo sampling.")
    parser.add_argument("--contingency_samples", type=int, default=1000,
                         help="Number of Monte Carlo forward simulations for contingency's "
                              "before/after pass-probability estimates.")
    parser.add_argument("--contingency_seed", type=int, default=None,
                         help="Seed for contingency's Monte Carlo sampling.")
    parser.add_argument("--contingency_split_seed", type=int, default=None,
                         help="Seed for split contingency Monte Carlo sampling "
                              "(defaults to --contingency_seed).")
    parser.add_argument("--csv", nargs="?", const="model_predictions.csv", default=None,
                         metavar="FILE",
                         help="Also save trial information to a CSV file "
                              "(default model_predictions.csv if no filename is given).")

    args = parser.parse_args()

    if args.trials_dir is None:
        args.trials_dir = "pilot_trials" if args.pilot else "trials"
    trials_dir = Path(args.trials_dir)

    if args.trial is not None:
        trial_dirs = [(str(args.trial), trials_dir / str(args.trial))]
        if not trial_dirs[0][1].is_dir():
            print(f"No such trial directory: {trial_dirs[0][1]}", file=sys.stderr)
            sys.exit(2)
    else:
        if not trials_dir.is_dir():
            print(f"No such trials directory: {trials_dir}", file=sys.stderr)
            sys.exit(2)
        trial_dirs = sorted(
            (d.name, d) for d in trials_dir.iterdir() if (d / "config.json").is_file()
        )
        if args.pilot:
            trial_dirs = [(name, d) for name, d in trial_dirs if name != "4"]
        if not trial_dirs:
            print(f"No trials found in {trials_dir}", file=sys.stderr)
            sys.exit(2)

    all_rows = []
    for trial_name, trial_dir in trial_dirs:
        all_rows.extend(_process_trial(trial_name, trial_dir, args))

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial", "agent", "day_signed", "pivotality",
                "pivotality_unlocked_later",
                "criticality_credit", "criticality_blame", "criticality_responsibility",
                "contingency_old_method",
                "contingency_can_sign_later",
                "contingency_split_never_signs",
                "contingency_simple",
                "contingency_shapley",
                "contingency_shapley_never_signs",
            ])
            writer.writerows(all_rows)
        print(f"Saved predictions to {args.csv}")


if __name__ == "__main__":
    main()
