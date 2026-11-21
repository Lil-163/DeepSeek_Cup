from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 180, "savefig.dpi": 220, "font.size": 10})

# Q1: aggregate test error by forecast horizon.
PUBLIC = ROOT / "outputs" / "paper_ready_20261122"
q1 = pd.read_csv(PUBLIC / "q1_horizon_error_summary.csv")
q1 = q1[q1["split"] == "test"]
labels = {
    "constant_velocity": "Constant velocity",
    "cv_kalman": "CV Kalman filter",
    "last_value": "Last observed value",
}
fig, ax = plt.subplots(figsize=(7.6, 4.8))
for method, label in labels.items():
    part = q1[q1["method"] == method].sort_values("horizon")
    ax.plot(part["horizon"], part["mean_error"], marker="o", linewidth=2, label=label)
ax.set(title="Q1: Test Forecast Error by Horizon", xlabel="Forecast horizon (frames)", ylabel="Mean position error")
ax.grid(True, alpha=.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "q1_test_error_by_horizon.png", bbox_inches="tight")
plt.close(fig)

# Q2: replay error and empirical coverage across mask durations.
q2 = pd.read_csv(PUBLIC / "q2_summary.csv")
method_labels = {
    "adaptive_request": "Adaptive request",
    "fixed_interval": "Fixed interval",
    "last_value": "Last value",
    "constant_velocity": "Constant velocity",
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
for method, label in method_labels.items():
    part = q2[q2["method"] == method].sort_values("duration_s")
    ax1.plot(part["duration_s"], part["mae"], marker="o", linewidth=1.8, label=label)
    if part["coverage_90"].notna().any():
        ax2.plot(part["duration_s"], part["coverage_90"], marker="o", linewidth=1.8, label=label)
ax1.set(title="Replay error", xlabel="Mask duration (s)", ylabel="Mean absolute error")
ax2.axhline(.90, color="black", linestyle="--", linewidth=1, label="Nominal 90%")
ax2.set(title="Empirical interval coverage", xlabel="Mask duration (s)", ylabel="Coverage")
for ax in (ax1, ax2):
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("Q2: Causal Replay Performance", y=1.03)
fig.tight_layout()
fig.savefig(OUT / "q2_replay_performance.png", bbox_inches="tight")
plt.close(fig)

# Q3: nominal-mapping validation on-time rate by case and policy.
q3 = pd.read_csv(PUBLIC / "q3_gate_0_50_nominal.csv")
fig, ax = plt.subplots(figsize=(8.0, 4.8))
policies = ["adaptive", "no_review", "edd"]
case_ids = ["S1", "S2", "S3"]
width = .24
colors = {"adaptive": "#3b82f6", "no_review": "#22a06b", "edd": "#8b93a1"}
for i, policy in enumerate(policies):
    part = q3[q3["policy"] == policy].set_index("case_id").reindex(case_ids)
    xs = [j + (i - 1) * width for j in range(len(case_ids))]
    ax.bar(xs, part["mean"], width=width, label=policy.replace("_", " "), color=colors[policy])
ax.axhline(.50, color="#b42318", linestyle="--", linewidth=1.4, label="0.50 threshold")
ax.set_xticks(range(len(case_ids)), case_ids)
ax.set_ylim(0, 1)
ax.set(title="Q3: Weighted On-Time Rate (Validation, Nominal Mapping)", xlabel="Scenario", ylabel="Weighted on-time rate")
ax.grid(axis="y", alpha=.25)
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "q3_nominal_on_time_rate.png", bbox_inches="tight")
plt.close(fig)

# Q3: empirical coverage of the duration quantile models.
qdur = pd.read_csv(PUBLIC / "q3_duration_validation.csv")
qdur = qdur[qdur["model"].isin(["q50", "q90"])]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
xs = range(len(qdur))
ax.bar([x - .18 for x in xs], qdur["nominal"], width=.36, label="Nominal coverage", color="#9aa4b2")
ax.bar([x + .18 for x in xs], qdur["coverage"], width=.36, label="Empirical coverage", color="#e07a5f")
ax.set_xticks(list(xs), qdur["model"].str.upper())
ax.set_ylim(0, 1)
ax.set(title="Q3: Duration Quantile Coverage", xlabel="Quantile model", ylabel="Coverage")
ax.grid(axis="y", alpha=.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "q3_duration_quantile_coverage.png", bbox_inches="tight")
plt.close(fig)

print(f"Created figures in {OUT}")
