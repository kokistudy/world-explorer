"""
Ways to produce a SimulationRun (the day-by-day record of who voted when).

  generate_random_spec    build a random NetworkSpec (agents + influence graph)
  simulate_trial          run a NetworkSpec forward with fresh random draws
  simulate_from_schedule  run a NetworkSpec forward, choosing random draws so
                           that voting happens on exactly the days requested
"""

from __future__ import annotations

import random
from pathlib import Path

from run_model import Agent, DayState, NetworkSpec, SimulationRun, update_thresholds

PROPENSITY_LEVELS = [0.1, 0.3, 0.5]
WEIGHT_LEVELS = [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

def load_names(names_file: str) -> list[str]:
    path = Path(names_file)
    if not path.exists():
        path = Path(__file__).parent / names_file
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def generate_random_spec(
    n_agents: int,
    n_days: int,
    pass_threshold: int,
    names_file: str,
    min_edges: int,
    max_edges: int,
    rng: random.Random,
) -> NetworkSpec:
    names_pool = load_names(names_file)
    names = rng.sample(names_pool, n_agents)

    # At least 5/9 of agents at the lowest propensity, at most 2/9 at the highest.
    n_high_max = max(0, round(n_agents * 2 / 9))
    n_low_min = max(0, round(n_agents * 5 / 9))
    n_high = rng.randint(0, n_high_max)
    remaining = n_agents - n_high
    n_low = rng.randint(min(n_low_min, remaining), remaining)
    n_mid = remaining - n_low
    levels = [PROPENSITY_LEVELS[0]] * n_low + [PROPENSITY_LEVELS[1]] * n_mid + [PROPENSITY_LEVELS[2]] * n_high
    rng.shuffle(levels)
    agents = [Agent(name=name, base_threshold=level) for name, level in zip(names, levels)]

    # Unidirectional edges only, capped at 3 outgoing / 2 incoming per agent, 10 total.
    max_out, max_in, max_total = 3, 1, 9
    candidate_pairs = [(a, b) for a in names for b in names if a != b]
    rng.shuffle(candidate_pairs)

    out_degree = {name: 0 for name in names}
    in_degree = {name: 0 for name in names}
    seen_pairs: set[frozenset] = set()
    edges: list[tuple[str, str]] = []

    n_edges_target = min(rng.randint(min_edges, max_edges), max_total)
    for src, dst in candidate_pairs:
        if len(edges) >= n_edges_target:
            break
        pair_key = frozenset((src, dst))
        if pair_key in seen_pairs:
            continue
        if out_degree[src] >= max_out or in_degree[dst] >= max_in:
            continue
        edges.append((src, dst))
        seen_pairs.add(pair_key)
        out_degree[src] += 1
        in_degree[dst] += 1

    # Majority of arrows should be the smallest weight (0.1).
    n_small = -(-len(edges) // 2)  # ceil(len(edges) / 2)
    weights = [WEIGHT_LEVELS[0]] * n_small + [rng.choice(WEIGHT_LEVELS[1:]) for _ in range(len(edges) - n_small)]
    rng.shuffle(weights)

    influence: dict[str, dict[str, float]] = {}
    for (src, dst), w in zip(edges, weights):
        influence.setdefault(src, {})[dst] = w

    return NetworkSpec(
        agents=agents,
        influence=influence,
        n_days=n_days,
        pass_threshold=pass_threshold,
    )


# ---------------------------------------------------------------------------
# Simulation: fresh random draws
# ---------------------------------------------------------------------------

def simulate_trial(spec: NetworkSpec, seed: int) -> SimulationRun:
    rng = random.Random(seed)

    thresholds = {a.name: a.base_threshold for a in spec.agents}
    voted_yes: set[str] = set()
    frozen_thresholds: dict[str, float] = {}
    day_voted: dict[str, int | None] = {a.name: None for a in spec.agents}
    days: list[DayState] = []

    for day in range(1, spec.n_days + 1):
        thresholds_start = dict(thresholds)
        draws = {name: round(rng.uniform(0, 1), 4) for name in thresholds}

        voted_yes_today = {
            name for name in thresholds
            if name not in voted_yes and draws[name] < thresholds[name]
        }
        for name in voted_yes_today:
            day_voted[name] = day
            frozen_thresholds[name] = thresholds_start[name]
        voted_yes |= voted_yes_today

        days.append(DayState(day=day, thresholds_start=thresholds_start, random_draws=draws))
        thresholds = update_thresholds(thresholds, voted_yes_today, spec)
        # Once an agent has voted, their propensity is frozen for the rest of the trial.
        thresholds.update(frozen_thresholds)

    return SimulationRun(spec=spec, seed=seed, days=days, day_voted=day_voted)


# ---------------------------------------------------------------------------
# Simulation: draws chosen to match a requested day_voted schedule
# ---------------------------------------------------------------------------

def simulate_from_schedule(
    spec: NetworkSpec,
    schedule: dict[str, int | None],
    seed: int,
) -> SimulationRun:
    """
    Run the model forward, picking each day's random draws so that every
    agent votes on exactly the day given in `schedule` (or never, if their
    entry is None/absent).

    Draws are sampled uniformly from the range consistent with that day's
    outcome: below the agent's current threshold on their voting day, at or
    above it on every day before that.

    Raises ValueError if the schedule is infeasible -- e.g. an agent is
    supposed to vote on a day when their threshold is 0, so no draw could
    ever be low enough.
    """
    rng = random.Random(seed)
    names = spec.agent_names
    unknown = set(schedule) - set(names)
    if unknown:
        raise ValueError(f"schedule references unknown agents: {sorted(unknown)}")
    target_day = {name: schedule.get(name) for name in names}

    thresholds = {a.name: a.base_threshold for a in spec.agents}
    voted_yes: set[str] = set()
    frozen_thresholds: dict[str, float] = {}
    day_voted: dict[str, int | None] = {name: None for name in names}
    days: list[DayState] = []

    for day in range(1, spec.n_days + 1):
        thresholds_start = dict(thresholds)
        draws: dict[str, float] = {}
        voted_yes_today = set()

        for name in names:
            if name in voted_yes:
                draws[name] = round(rng.uniform(0, 1), 4)
                continue

            th = thresholds_start[name]
            if target_day[name] == day:
                if th <= 0:
                    raise ValueError(
                        f"schedule wants {name} to sign on day {day}, but their "
                        f"threshold is {th} on that day -- impossible under this config"
                    )
                draws[name] = round(rng.uniform(0, th), 4)
                voted_yes_today.add(name)
            else:
                draws[name] = round(rng.uniform(th, 1.0), 4)

        for name in voted_yes_today:
            day_voted[name] = day
            frozen_thresholds[name] = thresholds_start[name]
        voted_yes |= voted_yes_today

        days.append(DayState(day=day, thresholds_start=thresholds_start, random_draws=draws))
        thresholds = update_thresholds(thresholds, voted_yes_today, spec)
        thresholds.update(frozen_thresholds)

    for name in names:
        if target_day[name] is not None and day_voted[name] != target_day[name]:
            raise ValueError(
                f"schedule wants {name} to sign on day {target_day[name]}, but they never got "
                f"the chance (n_days={spec.n_days})"
            )

    return SimulationRun(spec=spec, seed=seed, days=days, day_voted=day_voted)
