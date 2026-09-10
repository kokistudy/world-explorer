"""
Turn a SimulationRun into the day-by-day snapshot images.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch, FancyBboxPatch, Polygon

from run_model import NetworkSpec, SimulationRun, update_thresholds

BLUE_LIGHT, BLUE_DARK = "#a7bddf", "#186ded"
GREEN = "#1fbd53"
ARROW = "#c7cdd6"
FONT_FAMILY = "Open Sans"

CIRCLE_CENTER_Y = -0.5  # shifts the network down, away from the title text, while leaving margin at the bottom


def _lerp_color(low: str, high: str, t: float) -> tuple:
    lo = matplotlib.colors.to_rgb(low)
    hi = matplotlib.colors.to_rgb(high)
    t = max(0.0, min(1.0, t))
    return tuple(lo[i] + (hi[i] - lo[i]) * t for i in range(3))


def propensity_color(threshold: float, signed: bool) -> tuple:
    if signed:
        return matplotlib.colors.to_rgb(GREEN)
    lo, hi = 0.1, 0.8
    t = (threshold - lo) / (hi - lo)
    return _lerp_color(BLUE_LIGHT, BLUE_DARK, t)


def circle_layout(names: list[str]) -> dict[str, tuple]:
    n = len(names)
    positions = {}
    for i, name in enumerate(names):
        angle = math.pi / 2 - 2 * math.pi * i / n
        positions[name] = (4.5 * math.cos(angle), 4.5 * math.sin(angle) + CIRCLE_CENTER_Y)
    return positions


def draw_agent(ax, x, y, color, signed, threshold, mark=None):
    body = Polygon(
        [(x - 0.32, y - 0.64), (x + 0.32, y - 0.64), (x, y + 0.14)],
        closed=True, facecolor=color, edgecolor="none", zorder=3,
    )
    head = Circle((x, y + 0.115), 0.22, facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(body)
    ax.add_patch(head)

    bar_x = x + 0.48
    bar_bottom = y - 0.64
    bar_height = 0.8
    bar_width = 0.22
    ax.add_patch(FancyBboxPatch((bar_x, bar_bottom), bar_width, bar_height,
                                 boxstyle="round,pad=0,rounding_size=0.08",
                                 facecolor="#e4e8ee", edgecolor="none", zorder=2))
    fill_height = max(0.05, bar_height * max(0.0, min(1.0, threshold)))
    ax.add_patch(FancyBboxPatch((bar_x, bar_bottom), bar_width, fill_height,
                                 boxstyle="round,pad=0,rounding_size=0.08",
                                 facecolor=color, edgecolor="none", zorder=3))

    if signed and mark is None:
        check = ax.text(x, y - 0.37, "✓", ha="center", va="center",
                         fontsize=16, color="white", weight="bold", zorder=4)
        check.set_path_effects([
            path_effects.Stroke(linewidth=0.5, foreground="white"),
            path_effects.Normal(),
        ])
    elif signed:
        text = ax.text(x, y - 0.25, mark, ha="center", va="center",
                        fontsize=34, color="black", weight="bold", zorder=4)
        text.set_path_effects([
            path_effects.Stroke(linewidth=3.5, foreground="white"),
            path_effects.Normal(),
        ])


def render_day(spec: NetworkSpec, positions: dict, thresholds: dict, signed_by: set,
                day: int, n_days: int, needed: int, out_path: Path, day_voted: dict | None = None,
                title: str | None = None, show_signatures: bool = True):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # The propensity bar sits only to the right of each icon (icon reaches x-0.32
    # on the left, bar reaches x+0.70 on the right). Shift the arrow anchor point
    # by the midpoint of that asymmetry so a uniform shrink clears the icon on the
    # left and the bar on the right by the same margin.
    anchor_shift = ((0.48 + 0.22) - 0.32) / 2

    # The icon body (down to y-0.64) extends further below center than the head
    # (up to y+0.335) extends above it. Shift the anchor down by the midpoint of
    # that asymmetry so clearance above and below the icon ends up equal.
    anchor_shift_y = (0.335 - 0.64) / 2

    for src, dsts in spec.influence.items():
        for dst, weight in dsts.items():
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            arrow = FancyArrowPatch(
                (x0 + anchor_shift, y0 + anchor_shift_y), (x1 + anchor_shift, y1 + anchor_shift_y),
                arrowstyle=ArrowStyle("-|>", head_length=0.55, head_width=0.28),
                mutation_scale=14,
                shrinkA=32, shrinkB=32,
                linewidth=1 + weight ** 1.6 * 40,
                joinstyle="miter", capstyle="butt",
                color=ARROW, zorder=1, alpha=0.9,
            )
            ax.add_patch(arrow)

    for name, (x, y) in positions.items():
        signed = name in signed_by
        color = propensity_color(thresholds[name], signed)
        mark = str(day_voted[name]) if day_voted and signed else None
        draw_agent(ax, x, y, color, signed, thresholds[name], mark=mark)

        # Place the label radially outward from the circle's center, not to the
        # side, so chord-like influence arrows (which stay inside the circle) never
        # cross the name text. The propensity bar sits at a fixed offset to the
        # right of the icon (x+0.48 to x+0.70), so any label that isn't clearly
        # pulled to the left needs to clear it on the right; only clearly-left-
        # pulled labels are safe to sit flush against the icon's left side.
        #
        # Exception: the one or two agents nearest the bottom of the circle point
        # almost straight down (uy near -1), so their "radially outward" spot is
        # below the icon anyway -- put the label there directly, centered, instead
        # of off to a side.
        #
        # Exception: the agent at the very top points straight up, which would
        # place its label right against the "Day N" title text -- put it
        # directly above the icon, centered, instead.
        radius = math.hypot(x, y) or 1.0
        ux, uy = x / radius, y / radius
        offset = 0.6
        if uy < -0.85:
            label_x = x
            label_y = y - 1.15
            ha = "center"
        elif uy > 0.85:
            label_x = x
            label_y = y + 0.75
            ha = "center"
        else:
            pushed_left = ux < -0.3
            if pushed_left:
                label_x = x + min(ux * offset, -0.25)
                ha = "right"
            else:
                label_x = x + max(ux * offset, 0.98)
                ha = "left"
            label_y = y + uy * offset
        label = ax.text(label_x, label_y, name, ha=ha,
                         va="center", fontsize=17, weight="bold", family=FONT_FAMILY,
                         color="#1c2733", zorder=6)
        label.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ])

    title_text = title if title is not None else f"Day {day}"
    title_lines = [ln for ln in title_text.split("\n") if ln != ""]
    for i, line in enumerate(title_lines):
        ax.text(-6.8, 6.3 - 0.7 * i, line, fontsize=26 if i == 0 else 22, weight="bold",
                 color="#2f6fed", family=FONT_FAMILY)
    if show_signatures:
        sig_y = 5.5 - 0.7 * max(0, len(title_lines) - 1)
        ax.text(-6.8, sig_y, f"Signatures: {len(signed_by)}/{needed}", fontsize=20,
                 weight="bold", color="black", family=FONT_FAMILY)

    fig.savefig(out_path, dpi=125)
    plt.close(fig)


def _thresholds_by_day(run: SimulationRun) -> list[tuple[dict, set]]:
    """(thresholds, signed_by) after each day, starting with day 0 (before any votes)."""
    spec = run.spec
    thresholds = {a.name: a.base_threshold for a in spec.agents}
    signed_by: set = set()
    frozen_thresholds: dict[str, float] = {}
    states = [(dict(thresholds), set(signed_by))]

    for day_state in run.days:
        day = day_state.day
        signed_today = {name for name, d in run.day_voted.items() if d == day}
        for name in signed_today:
            frozen_thresholds[name] = thresholds[name]
        signed_by = signed_by | signed_today
        thresholds = update_thresholds(thresholds, signed_today, spec)
        # Once an agent has voted, their propensity is frozen for the rest of the trial.
        thresholds.update(frozen_thresholds)
        states.append((dict(thresholds), set(signed_by)))

    return states


def render_snapshots(run: SimulationRun, out_dir: Path):
    spec = run.spec
    positions = circle_layout(spec.agent_names)
    snapshots_dir = out_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    for day, (thresholds, signed_by) in enumerate(_thresholds_by_day(run)):
        render_day(spec, positions, thresholds, signed_by, day, spec.n_days,
                   spec.pass_threshold, snapshots_dir / f"day{day}.png")


def render_summary(run: SimulationRun, out_path: Path, trial_name: str):
    """Final day's snapshot, with each signed agent labeled by the day they signed."""
    spec = run.spec
    positions = circle_layout(spec.agent_names)
    thresholds, signed_by = _thresholds_by_day(run)[-1]
    render_day(spec, positions, thresholds, signed_by, spec.n_days, spec.n_days,
               spec.pass_threshold, out_path, day_voted=run.day_voted, title=f"Trial {trial_name}")


def render_summary_initial(run: SimulationRun, out_path: Path, trial_name: str):
    """Same as render_summary, but every bar is the agent's base (day-0) propensity."""
    spec = run.spec
    positions = circle_layout(spec.agent_names)
    thresholds = {a.name: a.base_threshold for a in spec.agents}
    signed_by = set(run.yes_voters)
    render_day(spec, positions, thresholds, signed_by, spec.n_days, spec.n_days,
               spec.pass_threshold, out_path, day_voted=run.day_voted, title=f"Trial {trial_name}")
