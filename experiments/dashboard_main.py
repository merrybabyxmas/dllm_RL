"""
Main Experiments Dashboard  --  cc_rl
Tracks: datasets x methods x gen_lengths

Run:
    streamlit run experiments/dashboard_main.py --server.port 8503
    # External access via serveo:
    ssh -R 80:localhost:8503 serveo.net
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
BASE         = PROJECT_ROOT / "experiments" / "outputs" / "main_experiments"
LOG_DIR      = BASE / "launcher_logs"

DATASETS     = ["mbpp", "humaneval", "svamp", "gsm8k", "countdown", "spider"]
METHODS      = ["baseline", "delta_v_only"]
GEN_LENGTHS  = [128, 256, 512]

COLORS = {
    "baseline":    "#888888",
    "delta_v_only": "#4C9BE8",
    "reward":      "#4C9BE8",
    "eval":        "#F4A62A",
    "policy":      "#E84C4C",
    "kl":          "#9B59B6",
}

_EVAL_RE = re.compile(r"\[EvalCallback step=(\d+)\]\s+mean_score=([\d\.]+)")
_STEP_RE = re.compile(
    r"'reward':\s*'([\d\.\-e]+)'.*?"
    r"'reward_std':\s*'([\d\.\-e]+)'.*?"
    r"'kl':\s*'([\d\.\-e]+)'"
)
_LOSS_RE = re.compile(r"'loss':\s*'([\d\.\-e]+)'")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def result_path(ds: str, gl: int, method: str) -> Path:
    return BASE / ds / f"gl{gl}" / method / "result.json"


def log_path(ds: str, gl: int, method: str) -> Path:
    return BASE / ds / f"gl{gl}" / method / "run.log"


def launcher_log_path(ds: str, gl: int, method: str) -> Path:
    return LOG_DIR / f"{ds}_gl{gl}_{method}.log"


def read_result(ds: str, gl: int, method: str) -> dict | None:
    p = result_path(ds, gl, method)
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def load_eval_history(ds: str, gl: int, method: str) -> pd.DataFrame:
    p = log_path(ds, gl, method)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text().splitlines():
        m = _EVAL_RE.search(line)
        if m:
            rows.append({"step": int(m.group(1)), "score": float(m.group(2))})
    return pd.DataFrame(rows)


def load_train_log(ds: str, gl: int, method: str) -> pd.DataFrame:
    p = launcher_log_path(ds, gl, method)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text().splitlines():
        m = _STEP_RE.search(line)
        if not m:
            continue
        row = {
            "step":    (len(rows) + 1) * 10,
            "reward":  float(m.group(1)),
            "kl":      float(m.group(3)),
        }
        lm = _LOSS_RE.search(line)
        if lm:
            row["policy_loss"] = float(lm.group(1))
        rows.append(row)
    return pd.DataFrame(rows)


def is_running(ds: str, gl: int, method: str) -> bool:
    """Check if the launcher log exists but result.json doesn't."""
    lp = launcher_log_path(ds, gl, method)
    return lp.exists() and not result_path(ds, gl, method).exists()


def get_last_log(ds: str, gl: int, method: str, n: int = 25) -> str:
    p = launcher_log_path(ds, gl, method)
    if not p.exists():
        return ""
    return "\n".join(p.read_text().splitlines()[-n:])


def gpu_info() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        lines = r.stdout.strip().splitlines()
        parts = []
        for line in lines:
            idx, mu, mt, util = [x.strip() for x in line.split(",")]
            mu_val = mu.replace(" MiB", "")
            mt_val = mt.replace(" MiB", "")
            parts.append(f"GPU{idx}: {mu_val}/{mt_val}MiB {util}")
        return " | ".join(parts)
    except Exception:
        return "nvidia-smi unavailable"


def gpu_per_card() -> list[dict]:
    """Returns list of per-GPU dicts."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu,name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        cards = []
        for line in r.stdout.strip().splitlines():
            idx, mu, mt, util, name = [x.strip() for x in line.split(",")]
            cards.append({
                "idx":   int(idx),
                "name":  name,
                "mem_used": int(mu.replace(" MiB", "")),
                "mem_total": int(mt.replace(" MiB", "")),
                "util":  int(util.replace(" %", "")),
            })
        return cards
    except Exception:
        return []


def smooth(series: pd.Series, w: int = 10) -> pd.Series:
    return series.rolling(w, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="cc_rl Main Experiments",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 0.8rem; padding-bottom: 0rem; }
  .metric-box { background:#1a1a2e; border-radius:6px; padding:8px 12px;
                margin:2px; font-size:0.82rem; }
  .done    { color:#4CE87A; font-weight:600; }
  .running { color:#F4A62A; font-weight:600; }
  .pending { color:#888888; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
hdr_left, hdr_right = st.columns([4, 2])
with hdr_left:
    st.markdown("## cc_rl Main Experiments Dashboard")
    st.caption(f"Base: `{BASE}`  |  Refresh: 30s  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
with hdr_right:
    st.code(gpu_info(), language=None)

st.divider()

# ---------------------------------------------------------------------------
# GPU status cards
# ---------------------------------------------------------------------------
gpu_cards = gpu_per_card()
if gpu_cards:
    st.markdown("### GPU Status (4x RTX 4090)")
    gcols = st.columns(len(gpu_cards))
    for gi, card in enumerate(gpu_cards):
        with gcols[gi]:
            pct = card["mem_used"] / card["mem_total"] * 100 if card["mem_total"] > 0 else 0
            st.metric(
                label=f"GPU {card['idx']} ({card['name']})",
                value=f"{card['util']}% util",
                delta=f"{card['mem_used']}/{card['mem_total']} MiB",
            )
            st.progress(pct / 100)
    st.divider()

# ---------------------------------------------------------------------------
# Results Summary Table
# ---------------------------------------------------------------------------
with st.expander("Results Summary Table", expanded=True):
    for gl in GEN_LENGTHS:
        st.markdown(f"**gen_length = {gl}**")
        rows = []
        for ds in DATASETS:
            row = {"dataset": ds}
            for method in METHODS:
                r = read_result(ds, gl, method)
                if r is not None:
                    row[method] = f"{r['mean_score']:.4f}"
                elif is_running(ds, gl, method):
                    row[method] = "running..."
                else:
                    row[method] = "---"
            # delta column
            baseline_r = read_result(ds, gl, "baseline")
            dv_r       = read_result(ds, gl, "delta_v_only")
            if baseline_r and dv_r:
                base_s = baseline_r["mean_score"]
                dv_s   = dv_r["mean_score"]
                delta  = (dv_s - base_s) / max(base_s, 1e-6) * 100
                row["delta_v vs baseline"] = f"{delta:+.1f}%"
            else:
                row["delta_v vs baseline"] = "---"
            rows.append(row)

        df = pd.DataFrame(rows).set_index("dataset")
        st.dataframe(df, use_container_width=True)
        st.markdown("")

st.divider()

# ---------------------------------------------------------------------------
# Per-dataset progress grid
# ---------------------------------------------------------------------------
st.markdown("### Per-Dataset Progress")

tab_labels = [f"gl={gl}" for gl in GEN_LENGTHS]
tabs = st.tabs(tab_labels)

for ti, gl in enumerate(GEN_LENGTHS):
    with tabs[ti]:
        cols = st.columns(len(DATASETS), gap="small")
        for ci, ds in enumerate(DATASETS):
            with cols[ci]:
                st.markdown(f"**{ds}**")
                for method in METHODS:
                    r        = read_result(ds, gl, method)
                    running  = is_running(ds, gl, method)
                    if r:
                        st.markdown(
                            f'<span class="done">{method}</span>: {r["mean_score"]:.4f}',
                            unsafe_allow_html=True,
                        )
                    elif running:
                        st.markdown(
                            f'<span class="running">{method}</span>: running',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<span class="pending">{method}</span>: pending',
                            unsafe_allow_html=True,
                        )

        st.divider()

        # Training curves for this gen_length
        st.markdown(f"**Training Curves  (gen_length={gl})**")
        ds_cols = st.columns(len(DATASETS), gap="small")
        for ci, ds in enumerate(DATASETS):
            with ds_cols[ci]:
                # Show eval history for delta_v_only (more interesting)
                eval_df = load_eval_history(ds, gl, "delta_v_only")
                train_df = load_train_log(ds, gl, "delta_v_only")

                fig = go.Figure()
                if not train_df.empty and "reward" in train_df.columns:
                    fig.add_trace(go.Scatter(
                        x=train_df["step"], y=smooth(train_df["reward"]),
                        mode="lines",
                        line=dict(color=COLORS["delta_v_only"], width=2),
                        name="reward (delta_v)",
                    ))
                if not eval_df.empty:
                    fig.add_trace(go.Scatter(
                        x=eval_df["step"], y=eval_df["score"],
                        mode="markers+lines",
                        marker=dict(size=5, color=COLORS["eval"]),
                        line=dict(color=COLORS["eval"], width=1.5, dash="dot"),
                        name="eval score",
                    ))

                # Final score hline
                r = read_result(ds, gl, "delta_v_only")
                if r:
                    fig.add_hline(y=r["mean_score"], line_dash="dash",
                                  line_color="white", opacity=0.5)

                fig.update_layout(
                    title=dict(text=ds, font=dict(size=12)),
                    height=200,
                    margin=dict(t=30, b=20, l=30, r=10),
                    paper_bgcolor="#1a1a2e",
                    plot_bgcolor="#16213e",
                    font=dict(color="#e0e0e0", size=10),
                    showlegend=False,
                )
                fig.update_xaxes(gridcolor="#2a2a4a")
                fig.update_yaxes(gridcolor="#2a2a4a")
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"curve_{ds}_gl{gl}")

st.divider()

# ---------------------------------------------------------------------------
# Live Log
# ---------------------------------------------------------------------------
st.markdown("### Live Log")

running_experiments = []
for ds in DATASETS:
    for method in METHODS:
        for gl in GEN_LENGTHS:
            if is_running(ds, gl, method):
                running_experiments.append((ds, gl, method))

if running_experiments:
    ds0, gl0, m0 = running_experiments[0]
    st.caption(f"Showing: **{ds0}/gl{gl0}/{m0}** -- last 30 lines")
    st.code(get_last_log(ds0, gl0, m0, 30), language=None)

    if len(running_experiments) > 1:
        with st.expander(f"Other running experiments ({len(running_experiments)-1})"):
            for ds, gl, method in running_experiments[1:]:
                st.caption(f"{ds}/gl{gl}/{method}")
                st.code(get_last_log(ds, gl, method, 10), language=None)
else:
    st.info("No active experiments detected.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)
