"""
Interactive toy-world explorer.

    cd code/exp4
    streamlit run explore_toy.py
"""
from __future__ import annotations

import inspect
import io
import tempfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# `python explore_toy.py` should start the website, not run the page once.
if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        import sys
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        raise SystemExit(stcli.main())

from render import _thresholds_by_day, circle_layout, render_day
from run_model import (
    Agent,
    NetworkSpec,
    compute_contingency,
    compute_contingency_shapley,
    compute_contingency_simple,
    compute_contingency_split,
    compute_criticality,
    compute_responsibility,
    compute_responsibility_unlocked_later,
    _simulate_with_draws,
    _draw_tail,
)

SCORE_CHOICES = [
    ("piv", "Pivotality", "Pivotality", True),
    ("piv_unlock", "Pivotality (later people can still change)", "Pivotality\n(later can change)", False),
    ("credit", "Criticality (credit)", "Criticality\n(credit)", True),
    ("blame", "Criticality (blame)", "Criticality\n(blame)", False),
    ("ant_resp", "Criticality (anticipated resp.)", "Criticality\n(ant. resp.)", False),
    ("simple", "Contingency (simple)", "Contingency\n(simple)", True),
    ("old", "Contingency (old method)", "Contingency\n(old method)", False),
    ("later", "Contingency (can sign later)", "Contingency\n(can sign later)", False),
    ("never", "Contingency (never signs)", "Contingency\n(never signs)", False),
    ("shapley", "Contingency (average over orders)", "Contingency\n(avg. over orders)", False),
    ("shapley_never", "Contingency (average over orders, never signs)", "Contingency\n(avg. orders, never)", False),
]
from simulate import simulate_from_schedule

PRESETS = {
    "World 1 · need 2 · A,B,C = 0.5": {
        "need": 2,
        "n_days": 1,
        "agents": [("A", 0.5), ("B", 0.5), ("C", 0.5)],
        "edges": [],
        "signed": {"A": 1, "B": 1, "C": 1},
    },
    "World 2 · need 3 · A,B,C = 0.5": {
        "need": 3,
        "n_days": 1,
        "agents": [("A", 0.5), ("B", 0.5), ("C", 0.5)],
        "edges": [],
        "signed": {"A": 1, "B": 1, "C": 1},
    },
    "World 3 · need 2 · 0.2 / 0.5 / 0.8": {
        "need": 2,
        "n_days": 1,
        "agents": [("A", 0.2), ("B", 0.5), ("C", 0.8)],
        "edges": [],
        "signed": {"A": 1, "B": 1, "C": 1},
    },
    "World 4 · need 3 · 0.2 / 0.8 / 0.8": {
        "need": 3,
        "n_days": 1,
        "agents": [("A", 0.2), ("B", 0.8), ("C", 0.8)],
        "edges": [],
        "signed": {"A": 1, "B": 1, "C": 1},
    },
    "Blank": {
        "need": 2,
        "n_days": 1,
        "agents": [("A", 0.5), ("B", 0.5)],
        "edges": [],
        "signed": {"A": 1, "B": 1},
    },
}


def _preset_frames(p):
    agents = pd.DataFrame(
        [{"name": n, "propensity": t} for n, t in p["agents"]]
    )
    edges = pd.DataFrame(p["edges"], columns=["from", "to", "weight"])
    if edges.empty:
        edges = pd.DataFrame(columns=["from", "to", "weight"])
    signed = pd.DataFrame(
        [
            {"name": n, "day_signed": p["signed"].get(n)}
            for n, _ in p["agents"]
        ]
    )
    return agents, edges, signed, p["need"], p["n_days"]


def _spec_from_editors(agents: pd.DataFrame, edges: pd.DataFrame, n_days: int, need: int):
    agents = agents.dropna(subset=["name"])
    names = []
    seen = set()
    people = []
    for _, row in agents.iterrows():
        name = str(row["name"]).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        people.append(Agent(name=name, base_threshold=float(row["propensity"])))
    influence: dict[str, dict[str, float]] = {}
    if len(edges):
        for _, row in edges.dropna(subset=["from", "to"]).iterrows():
            src, dst = str(row["from"]).strip(), str(row["to"]).strip()
            if src not in seen or dst not in seen or src == dst:
                continue
            w = float(row["weight"])
            influence.setdefault(src, {})[dst] = w
    spec = NetworkSpec(
        agents=people,
        influence=influence,
        n_days=int(n_days),
        pass_threshold=int(need),
    )
    return spec, names


def _schedule(signed: pd.DataFrame, names: list[str], n_days: int):
    by_name = {}
    if len(signed):
        for _, row in signed.iterrows():
            name = str(row.get("name", "")).strip()
            raw = row.get("day_signed")
            if name == "" or pd.isna(raw) or raw == "" or raw is None:
                by_name[name] = None
            else:
                day = int(raw)
                by_name[name] = day if 1 <= day <= n_days else None
    return {n: by_name.get(n) for n in names}


def _p_pass(spec: NetworkSpec, n: int, seed: int) -> float:
    import numpy as np

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        draws = _draw_tail(spec, 1, rng)
        voted = _simulate_with_draws(spec, draws)
        hits += sum(v is not None for v in voted.values()) >= spec.pass_threshold
    return hits / n


def _draw_network(run, p_pass: float) -> bytes:
    spec = run.spec
    positions = circle_layout(spec.agent_names)
    thresholds, signed_by = _thresholds_by_day(run)[-1]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "net.png"
        kwargs = dict(
            spec=spec, positions=positions, thresholds=thresholds,
            signed_by=signed_by, day=spec.n_days, n_days=spec.n_days,
            needed=spec.pass_threshold, out_path=path,
            day_voted=run.day_voted,
            title=(
                f"Need {spec.pass_threshold}  ·  {spec.n_days} day(s)\n"
                f"P(pass) before anyone signs: {p_pass:.3f}"
            ),
        )
        if "show_signatures" in inspect.signature(render_day).parameters:
            kwargs["show_signatures"] = True
        render_day(**kwargs)
        return path.read_bytes()


def _wrap_header(label: str) -> str:
    label = " ".join(label.split())
    if "(" in label and label.endswith(")"):
        main, rest = label.split("(", 1)
        main = main.strip()
        paren = "(" + rest
        if len(paren) > 28:
            inner = rest[:-1] if rest.endswith(")") else rest
            wrapped = textwrap.wrap(inner, width=26)
            paren = "(" + "\n".join(wrapped) + ")"
        return f"{main}\n{paren}" if main else paren
    if len(label) > 16:
        return "\n".join(textwrap.wrap(label, width=14))
    return label


def _table_png(rows: list[dict], columns: list[tuple[str, str]]) -> bytes:
    n_people = max(1, len(rows))
    n_cols = max(1, len(columns))
    keys = [k for k, _ in columns]
    headers = [_wrap_header(h) for _, h in columns]
    n_lines = max(h.count("\n") + 1 for h in headers)
    fig_w = max(10.0, 2.2 * n_cols)
    fig_h = 1.2 + 0.42 * n_people + 0.22 * n_lines
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    cell = [[row[k] for k in keys] for row in rows]
    table = ax.table(
        cellText=cell, colLabels=headers, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(n_cols)))
    table.scale(1.0, 2.2)
    for (r, c), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor("#d0d5dd")
        cell_obj.PAD = 0.08
        if r == 0:
            cell_obj.set_facecolor("#eef2f6")
            cell_obj.set_text_props(weight="bold")
            cell_obj.set_height(0.10 + 0.08 * n_lines)
        else:
            cell_obj.set_facecolor("white")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


st.set_page_config(page_title="Toy world explorer", layout="wide")
st.title("Toy world explorer")
st.caption(
    "Edit the network and who signed. Scores use the same functions as the "
    "experiment model: pivotality, criticality (credit), simple contingency."
)

if "editor_nonce" not in st.session_state:
    st.session_state.editor_nonce = 0
if "preset_name" not in st.session_state:
    st.session_state.preset_name = "World 1 · need 2 · A,B,C = 0.5"
    a, e, s, need, nd = _preset_frames(PRESETS[st.session_state.preset_name])
    st.session_state.agents = a
    st.session_state.edges = e
    st.session_state.signed = s
    st.session_state.need = need
    st.session_state.n_days = nd

with st.sidebar:
    st.header("World")
    pick = st.selectbox("Load a starting world", list(PRESETS), index=0)
    if st.button("Load"):
        st.session_state.preset_name = pick
        a, e, s, need, nd = _preset_frames(PRESETS[pick])
        st.session_state.agents = a
        st.session_state.edges = e
        st.session_state.signed = s
        st.session_state.need = need
        st.session_state.n_days = nd
        st.session_state.editor_nonce += 1
        st.rerun()

    st.session_state.need = st.number_input(
        "Signatures needed", min_value=1, max_value=20,
        value=int(st.session_state.need),
    )
    st.session_state.n_days = st.number_input(
        "Number of days", min_value=1, max_value=10,
        value=int(st.session_state.n_days),
    )
    n_samples = st.slider("Simulations", 200, 4000, 4000, step=200)
    seed = st.number_input("Seed", min_value=0, value=1)
    st.subheader("Scores")
    st.caption("The first three match the explorer default. Turn on any extra pilot scores.")
    selected = []
    for key, label, _header, default in SCORE_CHOICES:
        if st.checkbox(label, value=default, key=f"score_{key}"):
            selected.append(key)

left, right = st.columns([1, 1.15])

with left:
    st.subheader("People")
    st.session_state.agents = st.data_editor(
        st.session_state.agents,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name", required=True),
            "propensity": st.column_config.NumberColumn(
                "Initial propensity", min_value=0.0, max_value=1.0,
                step=0.1, format="%.2f",
            ),
        },
        key=f"agents_ed_{st.session_state.editor_nonce}",
    )
    st.subheader("Arrows")
    st.caption("Leave empty for no influence. From A to B with weight 0.4 means A → B.")
    st.session_state.edges = st.data_editor(
        st.session_state.edges,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "from": st.column_config.TextColumn("From"),
            "to": st.column_config.TextColumn("To"),
            "weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, max_value=1.0, step=0.1, format="%.2f",
            ),
        },
        key=f"edges_ed_{st.session_state.editor_nonce}",
    )
    st.subheader("Who signed")
    st.caption("Day signed is 1…N, or blank if they never signed.")
    names_now = [
        str(r["name"]).strip()
        for _, r in st.session_state.agents.dropna(subset=["name"]).iterrows()
        if str(r["name"]).strip()
    ]
    have = (
        set(st.session_state.signed["name"].astype(str))
        if len(st.session_state.signed)
        else set()
    )
    extra = [n for n in names_now if n not in have]
    if extra:
        add = pd.DataFrame([{"name": n, "day_signed": None} for n in extra])
        st.session_state.signed = pd.concat(
            [st.session_state.signed, add], ignore_index=True
        )
    st.session_state.signed = st.data_editor(
        st.session_state.signed,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "day_signed": st.column_config.NumberColumn(
                "Day signed", min_value=1, max_value=10, step=1,
            ),
        },
        key=f"signed_ed_{st.session_state.editor_nonce}",
    )
    run_btn = st.button("Compute scores", type="primary")

with right:
    if run_btn:
        try:
            spec, names = _spec_from_editors(
                st.session_state.agents,
                st.session_state.edges,
                st.session_state.n_days,
                st.session_state.need,
            )
            if not names:
                st.error("Add at least one person.")
            elif spec.pass_threshold > len(names):
                st.error("Need more signatures than there are people.")
            else:
                schedule = _schedule(
                    st.session_state.signed, names, spec.n_days
                )
                run = simulate_from_schedule(spec, schedule, int(seed))
                p_pass = _p_pass(spec, int(n_samples), int(seed))
                n, sd = int(n_samples), int(seed)
                want = set(selected)
                scores = {name: {} for name in names}
                if want & {"piv"}:
                    got = compute_responsibility(run, n_samples=n, seed=sd)
                    for name in names:
                        scores[name]["piv"] = got[name]
                if want & {"piv_unlock"}:
                    got = compute_responsibility_unlocked_later(
                        run, n_samples=n, seed=sd
                    )
                    for name in names:
                        scores[name]["piv_unlock"] = got[name]
                if want & {"credit", "blame", "ant_resp"}:
                    got = compute_criticality(spec, n_samples=n, seed=sd)
                    for name in names:
                        scores[name]["credit"] = got[name]["credit"]
                        scores[name]["blame"] = got[name]["blame"]
                        scores[name]["ant_resp"] = got[name]["responsibility"]
                if want & {"simple"}:
                    got = compute_contingency_simple(run, n_samples=n, seed=sd)
                    for name in names:
                        scores[name]["simple"] = got[name]
                if want & {"old"}:
                    got = compute_contingency(run, n_samples=n, seed=sd)
                    for name in names:
                        scores[name]["old"] = got[name]
                if want & {"later"}:
                    got = compute_contingency_split(
                        run, n_samples=n, seed=sd, never_signs=False
                    )
                    for name in names:
                        scores[name]["later"] = got[name]
                if want & {"never"}:
                    got = compute_contingency_split(
                        run, n_samples=n, seed=sd, never_signs=True
                    )
                    for name in names:
                        scores[name]["never"] = got[name]
                if want & {"shapley"}:
                    got = compute_contingency_shapley(
                        run, n_samples=n, seed=sd, never_signs=False
                    )
                    for name in names:
                        scores[name]["shapley"] = got[name]
                if want & {"shapley_never"}:
                    got = compute_contingency_shapley(
                        run, n_samples=n, seed=sd, never_signs=True
                    )
                    for name in names:
                        scores[name]["shapley_never"] = got[name]

                columns = [
                    ("Person", "Person"),
                    ("Initial propensity", "Initial\npropensity"),
                    ("Day signed", "Day\nsigned"),
                ]
                for key, label, _header, _default in SCORE_CHOICES:
                    if key in want:
                        columns.append((key, label))

                rows = []
                prop = {a.name: a.base_threshold for a in spec.agents}
                for name in names:
                    d = run.day_voted[name]
                    row = {
                        "Person": name,
                        "Initial propensity": f"{prop[name]:.2f}",
                        "Day signed": "—" if d is None else str(d),
                    }
                    for key in want:
                        row[key] = f"{scores[name][key]:.3f}"
                    rows.append(row)
                stem = f"toy_need{spec.pass_threshold}_{spec.n_days}d"
                st.session_state.structure_png = _draw_network(run, p_pass)
                st.session_state.table_png = _table_png(rows, columns)
                st.session_state.structure_name = f"{stem}_structure.png"
                st.session_state.table_name = f"{stem}_table.png"
        except ValueError as err:
            st.error(str(err))
    if st.session_state.get("structure_png") and st.session_state.get("table_png"):
        st.subheader("Structure")
        st.image(st.session_state.structure_png, use_container_width=True)
        st.download_button(
            "Download structure",
            data=st.session_state.structure_png,
            file_name=st.session_state.structure_name,
            mime="image/png",
            key="dl_structure",
        )
        st.subheader("Table")
        st.image(st.session_state.table_png, use_container_width=True)
        st.download_button(
            "Download table",
            data=st.session_state.table_png,
            file_name=st.session_state.table_name,
            mime="image/png",
            key="dl_table",
        )
    elif not run_btn:
        st.info("Set up the world on the left, then click Compute scores.")
