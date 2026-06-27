"""
Real-time experiment dashboard.

Layout:
  - Results table: baseline / diffu_grpo / stage2-v2 / stage2-v4
  - Grid: diffu_grpo × 3 datasets
  - Grid: stage2 v2 (max_value_states=2) × 3 datasets
  - Grid: stage2 v4 (max_value_states=4) × 3 datasets  ← NEW
  - Live log: first running experiment found

Run:
  streamlit run experiments/dashboard.py --server.port 8502
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
BASE_V2  = Path(__file__).parent.parent / "experiments" / "outputs" / "official_9exp"
BASE_V4  = Path(__file__).parent.parent / "experiments" / "outputs" / "official_9exp_v4"
BASE_AB  = Path(__file__).parent.parent / "experiments" / "outputs" / "official_9exp_ablation"
DATASETS = ["mbpp", "humaneval", "svamp"]
COLORS   = {
    "reward":  "#4C9BE8",
    "eval":    "#F4A62A",
    "policy":  "#E84C4C",
    "value":   "#4CE87A",
    "kl":      "#9B59B6",
}

_STEP_RE = re.compile(
    r"'reward':\s*'([\d\.\-e]+)'.*?"
    r"'reward_std':\s*'([\d\.\-e]+)'.*?"
    r"'kl':\s*'([\d\.\-e]+)'.*?"
    r"'step_time':\s*'([\d\.]+)'"
)
_LOSS_RE = re.compile(r"'loss':\s*'([\d\.\-e]+)'")
_VAL_RE  = re.compile(r"'value_loss':\s*'([\d\.\-e]+)'")
_EVAL_RE = re.compile(r"\[EvalCallback step=(\d+)\]\s+mean_score=([\d\.]+)")
_TQDM_RE = re.compile(r"(\d+)/(\d+)\s+\[[\d:]+<([\d:]+),\s*([\d\.]+)s/it\]")

# ---------------------------------------------------------------------------
def read_result(base, ds, method):
    p = base / ds / method / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_train_log(base, ds, method):
    p = base / f"{ds}_{method}.log"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text().splitlines():
        m = _STEP_RE.search(line)
        if not m:
            continue
        row = {
            "step":       (len(rows) + 1) * 10,
            "reward":     float(m.group(1)),
            "reward_std": float(m.group(2)),
            "kl":         float(m.group(3)),
            "step_time":  float(m.group(4)),
        }
        lm = _LOSS_RE.search(line)
        if lm:
            row["policy_loss"] = float(lm.group(1))
        vm = _VAL_RE.search(line)
        if vm:
            row["value_loss"] = float(vm.group(1))
        rows.append(row)
    return pd.DataFrame(rows)


def load_eval_history(base, ds, method):
    p = base / ds / method / "run.log"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text().splitlines():
        m = _EVAL_RE.search(line)
        if m:
            rows.append({"step": int(m.group(1)), "score": float(m.group(2))})
    return pd.DataFrame(rows)


def get_progress(base, ds, method):
    p = base / f"{ds}_{method}.log"
    if not p.exists():
        return None
    for line in reversed(p.read_text().splitlines()):
        m = _TQDM_RE.search(line)
        if m:
            done  = int(m.group(1))
            total = int(m.group(2))
            eta_str = m.group(3)
            parts = list(map(int, eta_str.split(":")))
            eta_s = parts[-1] + parts[-2]*60 + (parts[-3]*3600 if len(parts)==3 else 0)
            step_s = float(m.group(4))
            return done, total, eta_s, step_s
    return None


def get_last_log(base, ds, method, n=30):
    p = base / f"{ds}_{method}.log"
    if not p.exists():
        return ""
    return "\n".join(p.read_text().splitlines()[-n:])


def gpu_info():
    import subprocess
    r = subprocess.run(
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader",
        shell=True, capture_output=True, text=True,
    )
    return r.stdout.strip()


def smooth(series, w=10):
    return series.rolling(w, min_periods=1).mean()


def make_cell_figure(base, ds, method):
    df      = load_train_log(base, ds, method)
    eval_df = load_eval_history(base, ds, method)
    r       = read_result(base, ds, method)
    is_stage2 = method == "stage2"
    has_value = is_stage2 and not df.empty and "value_loss" in df.columns

    specs = [
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ]
    subplot_titles = [
        "Reward", "Policy Loss",
        "KL Divergence", "Value Loss" if is_stage2 else "",
    ]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    if df.empty:
        fig.add_annotation(
            text="No data yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=18, color="gray"),
        )
        fig.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20),
                          paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
                          font=dict(color="#e0e0e0"))
        return fig

    fig.add_trace(go.Scatter(
        x=df["step"], y=df["reward"],
        mode="lines", opacity=0.2, line=dict(color=COLORS["reward"]), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["step"], y=smooth(df["reward"]),
        mode="lines", line=dict(color=COLORS["reward"], width=2), showlegend=False,
    ), row=1, col=1)
    if not eval_df.empty:
        fig.add_trace(go.Scatter(
            x=eval_df["step"], y=eval_df["score"],
            mode="markers+lines",
            marker=dict(size=6, color=COLORS["eval"]),
            line=dict(color=COLORS["eval"], width=1.5, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
    if r:
        fig.add_hline(y=r["mean_score"], line_dash="dash",
                      line_color="white", opacity=0.5, row=1, col=1)

    if "policy_loss" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["step"], y=df["policy_loss"],
            mode="lines", opacity=0.15, line=dict(color=COLORS["policy"]), showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=df["step"], y=smooth(df["policy_loss"]),
            mode="lines", line=dict(color=COLORS["policy"], width=2), showlegend=False,
        ), row=1, col=2)

    if "kl" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["step"], y=smooth(df["kl"]),
            mode="lines", line=dict(color=COLORS["kl"], width=2), showlegend=False,
        ), row=2, col=1)

    if has_value:
        fig.add_trace(go.Scatter(
            x=df["step"], y=smooth(df["value_loss"]),
            mode="lines", line=dict(color=COLORS["value"], width=2), showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        height=340,
        margin=dict(t=40, b=10, l=30, r=10),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=11),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor="#2a2a4a", zerolinecolor="#2a2a4a")
    return fig


def render_cell_header(base, ds, method):
    r    = read_result(base, ds, method)
    prog = get_progress(base, ds, method)
    if r:
        baseline_r = read_result(BASE_V2, ds, "baseline")
        delta = ""
        if baseline_r:
            d = (r["mean_score"] - baseline_r["mean_score"]) / max(baseline_r["mean_score"], 1e-6) * 100
            delta = f"  ({d:+.1f}% vs base)"
        st.markdown(
            f'<div class="cell-header"><span class="status-done">✅ {ds}</span>'
            f' — <b>{r["mean_score"]:.4f}</b>{delta}</div>',
            unsafe_allow_html=True,
        )
    elif prog:
        done, total, eta_s, step_s = prog
        pct = done / total * 100
        eta_min = eta_s // 60
        st.markdown(
            f'<div class="cell-header"><span class="status-running">🔄 {ds}</span>'
            f' — {done}/{total} ({pct:.0f}%)  ETA {eta_min}m  {step_s:.1f}s/step</div>',
            unsafe_allow_html=True,
        )
        st.progress(pct / 100)
    else:
        st.markdown(
            f'<div class="cell-header"><span class="status-waiting">⏳ {ds}</span>'
            f' — not started</div>',
            unsafe_allow_html=True,
        )


def render_grid(label, base, method, key_prefix):
    st.markdown(f"### {label}")
    cols = st.columns(3, gap="small")
    for ci, ds in enumerate(DATASETS):
        with cols[ci]:
            render_cell_header(base, ds, method)
            fig = make_cell_figure(base, ds, method)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"{key_prefix}_{ds}")
    st.divider()


# ===========================================================================
# Page
# ===========================================================================

st.set_page_config(
    page_title="cc_rl Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 0rem; }
  .cell-header { font-size: 0.85rem; font-weight: 600; margin-bottom: 2px; }
  .status-done    { color: #4CE87A; }
  .status-running { color: #F4A62A; }
  .status-waiting { color: #888888; }
</style>
""", unsafe_allow_html=True)

# ── Top bar ──────────────────────────────────────────────────────────────────
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown("## 🧪 cc_rl Experiment Dashboard")
with top_right:
    st.markdown(f"**GPU** `{gpu_info()}`")
    st.caption(time.strftime("%H:%M:%S"))

# ── Results table ─────────────────────────────────────────────────────────────
with st.expander("📊 Results Table", expanded=True):
    def fmt(r, baseline=None):
        if r is None:
            return "—"
        s = f"{r['mean_score']:.4f}"
        if baseline and baseline.get("mean_score"):
            d = (r["mean_score"] - baseline["mean_score"]) / max(baseline["mean_score"], 1e-6) * 100
            s += f" ({d:+.1f}%)"
        return s

    rows = []
    for ds in DATASETS:
        baseline_r = read_result(BASE_V2, ds, "baseline")
        rows.append({
            "dataset":           ds,
            "baseline":          fmt(baseline_r),
            "diffu_grpo":        fmt(read_result(BASE_V2, ds, "diffu_grpo"),   baseline_r),
            "cw_grpo 🆕":        fmt(read_result(BASE_AB, ds, "cw_grpo"),      baseline_r),
            "delta_v_only 🆕":   fmt(read_result(BASE_AB, ds, "delta_v_only"), baseline_r),
            "stage2 v4 🆕":      fmt(read_result(BASE_V4, ds, "stage2"),       baseline_r),
        })

    st.dataframe(pd.DataFrame(rows).set_index("dataset"), use_container_width=True)

st.divider()

# ── Grids ────────────────────────────────────────────────────────────────────
render_grid("Diffu-GRPO  (baseline method)",          BASE_V2, "diffu_grpo",   "dg")
render_grid("CW-GRPO  (confidence weight only)",      BASE_AB, "cw_grpo",      "cw")
render_grid("Delta-V only  (no confidence weight)",   BASE_AB, "delta_v_only", "dv")
render_grid("Stage 2  —  delta-V + CW (max_states=4)", BASE_V4, "stage2",     "s2v4")

# ── Live log ─────────────────────────────────────────────────────────────────
st.markdown("### 📋 Live Log")

# Priority: ablation > v4 > v2
WATCH_ORDER = [
    (BASE_AB, "cw_grpo",      "cw_grpo"),
    (BASE_AB, "delta_v_only", "delta_v_only"),
    (BASE_V4, "stage2",       "stage2-v4"),
    (BASE_V2, "diffu_grpo",   "diffu_grpo"),
]
found = False
for base, method, label in WATCH_ORDER:
    for ds in DATASETS:
        if not read_result(base, ds, method) and get_progress(base, ds, method):
            st.caption(f"**{ds} / {label}** — last 30 lines")
            st.code(get_last_log(base, ds, method, 30), language=None)
            found = True
            break
    if found:
        break

if not found:
    st.info("No active experiments detected.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)
