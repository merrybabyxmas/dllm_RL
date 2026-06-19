"""
300-step checkpoint monitor with matplotlib graphs.
Run: python monitor_plot.py
Prints summary + saves PNG whenever either experiment crosses a new 300-step mark.
"""
import json, datetime, subprocess, sys, os, time
from pathlib import Path

LOGS = {
    "diffu_grpo": "experiments/outputs/diffu_grpo_3000step/train.log",
    "stage2":     "experiments/outputs/stage2_3000step/train.log",
}
COLORS = {"diffu_grpo": "#2196F3", "stage2": "#FF5722"}
LABELS = {"diffu_grpo": "Diffu-GRPO (baseline)", "stage2": "Stage 2 (value credit)"}
OUT_DIR = Path("experiments/outputs/monitor_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── data loading ──────────────────────────────────────────────────────────────
def load_steps(path):
    steps = []
    try:
        with open(path) as f:
            for line in f:
                l = line.strip()
                if l.startswith('{"step"'):
                    try: steps.append(json.loads(l))
                    except: pass
    except FileNotFoundError:
        pass
    return steps

# ── text summary ──────────────────────────────────────────────────────────────
def text_summary(key, steps, milestone):
    if not steps:
        print(f"  [{LABELS[key]}] no data yet")
        return
    last = steps[-1]
    n = len(steps)
    recent = steps[-30:]
    rewards = [s["mean_reward"] for s in recent]
    ng = sum(1 for s in recent if s["reward_std"] > 0.01)
    eta = (3000 - last["step"]) * last["step_time_s"] / 3600
    bar_len = 40
    filled = int(last["step"] / 3000 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  ┌─ {LABELS[key]}")
    print(f"  │  [{bar}] {last['step']}/3000 ({last['step']/30:.1f}%)  ETA {eta:.1f}h")
    print(f"  │  reward  : {sum(rewards)/len(rewards):.3f} (recent {len(recent)})  last={last['mean_reward']:.3f}±{last['reward_std']:.3f}")
    print(f"  │  loss    : {last['loss']:.4f}   clip={last['clip_fraction']:.4f}   kl={last['mean_kl']:.4f}")
    print(f"  │  grad    : {ng}/{len(recent)} steps with reward_std>0")
    if last.get("value_loss", 0) != 0:
        print(f"  │  val_loss: {last['value_loss']:.6f}   expvar={last['explained_var']:.3f}   conf={last['mean_confidence']:.3f}")
    print(f"  └  speed   : {last['step_time_s']:.0f}s/step")

# ── plot ──────────────────────────────────────────────────────────────────────
def make_plots(all_steps, milestone):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Training Monitor — Step {milestone} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    metrics = [
        ("mean_reward",    "Reward",          gs[0, 0]),
        ("loss",           "Loss",             gs[0, 1]),
        ("clip_fraction",  "Clip Fraction",    gs[1, 0]),
        ("mean_kl",        "KL Divergence",    gs[1, 1]),
        ("reward_std",     "Reward Std (diversity)", gs[2, 0]),
        ("value_loss",     "Value Loss (Stage 2)",   gs[2, 1]),
    ]

    for metric_key, title, gs_pos in metrics:
        ax = fig.add_subplot(gs_pos)
        plotted = False
        for key, steps in all_steps.items():
            if not steps:
                continue
            xs = [s["step"] for s in steps]
            ys = [s.get(metric_key, 0) for s in steps]
            if any(y != 0 for y in ys) or metric_key not in ("value_loss",):
                ax.plot(xs, ys, color=COLORS[key], label=LABELS[key],
                        linewidth=1.2, alpha=0.9)
                plotted = True
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=7, loc="best")
        # Mark 300-step milestones
        for m in range(300, 3001, 300):
            ax.axvline(x=m, color="gray", linestyle="--", alpha=0.2, linewidth=0.8)

    fname = OUT_DIR / f"step_{milestone:05d}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    return str(fname)

# ── watcher loop ─────────────────────────────────────────────────────────────
def watch(one_shot=False, trigger_step=None):
    last_milestone = {k: 0 for k in LOGS}
    poll_interval = 60  # seconds

    # If trigger_step given, just plot that milestone now
    if trigger_step is not None:
        all_steps = {k: load_steps(v) for k, v in LOGS.items()}
        gpu = subprocess.run(
            "nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader",
            shell=True, capture_output=True, text=True).stdout.strip()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*62}")
        print(f"  Checkpoint @ step {trigger_step}  |  {ts}  |  GPU: {gpu}")
        print(f"{'='*62}")
        for key, steps in all_steps.items():
            text_summary(key, steps, trigger_step)
        fname = make_plots(all_steps, trigger_step)
        print(f"\n  Plot saved → {fname}")
        return

    print(f"[monitor] Watching logs, polling every {poll_interval}s. Reporting every 300 steps.")
    while True:
        fired = False
        all_steps = {k: load_steps(v) for k, v in LOGS.items()}
        for key, steps in all_steps.items():
            if not steps:
                continue
            cur = steps[-1]["step"]
            milestone = (cur // 300) * 300
            if milestone > 0 and milestone > last_milestone[key]:
                last_milestone[key] = milestone
                fired = True
        if fired:
            milestone = max(last_milestone.values())
            gpu = subprocess.run(
                "nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader",
                shell=True, capture_output=True, text=True).stdout.strip()
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n{'='*62}")
            print(f"  Checkpoint @ step {milestone}  |  {ts}  |  GPU: {gpu}")
            print(f"{'='*62}")
            for key, steps in all_steps.items():
                text_summary(key, steps, milestone)
            fname = make_plots(all_steps, milestone)
            print(f"\n  Plot saved → {fname}")
            sys.stdout.flush()
        if one_shot:
            break
        time.sleep(poll_interval)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=None, help="Plot at this specific step immediately")
    ap.add_argument("--once", action="store_true", help="Single poll then exit")
    args = ap.parse_args()
    watch(one_shot=args.once, trigger_step=args.step)
