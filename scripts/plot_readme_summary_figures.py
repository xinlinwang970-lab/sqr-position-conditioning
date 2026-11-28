"""Generate the compact summary figures used by the project README.

The charts read committed manifests and result JSON files directly so the
displayed values remain traceable to the release artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"
OUTPUT = ROOT / "figures"

BLUE = "#31688E"
ORANGE = "#D95F02"
GREEN = "#2A9D8F"
PURPLE = "#7A5195"
GRID = "#D7DCE2"
TEXT = "#20252B"
MUTED = "#5D6670"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": "#8C959F",
        "axes.titlecolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
    }
)


def read_final(relative_path: str) -> dict[str, float]:
    path = RESULTS / relative_path
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    final = data["final"]
    if final["n"] != 48:
        raise ValueError(f"Expected 48 test queries in {path}, found {final['n']}")
    return final


def read_history(run_name: str) -> list[dict[str, float]]:
    path = RESULTS / run_name / "val_history.json"
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    epochs = [row["epoch"] for row in data]
    if len(data) != 40 or epochs[-1] != 999:
        raise ValueError(f"Expected 40 validation points through epoch 999 in {path}")
    if epochs != sorted(epochs) or any(row["n"] != 48 for row in data):
        raise ValueError(f"Invalid validation history in {path}")
    return data


def read_manifest(name: str) -> list[dict[str, object]]:
    path = MANIFESTS / name
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=200, facecolor="white", bbox_inches="tight")
    svg_path = OUTPUT / f"{stem}.svg"
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def style_numeric_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#9BA3AA", linewidth=0.9)


def style_validation_axis(
    ax: plt.Axes,
    *,
    y_limits: tuple[float, float],
    x_max: float = 1050,
) -> None:
    ax.set_xlim(0, x_max)
    ax.set_ylim(*y_limits)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#9BA3AA", linewidth=0.9)


def add_validation_curve(
    ax: plt.Axes,
    history: list[dict[str, float]],
    *,
    key: str,
    label: str,
    color: str,
    linestyle: str,
    marker: str,
) -> tuple[Line2D, float, float]:
    epochs = [row["epoch"] for row in history]
    values = [row[key] for row in history]
    (line,) = ax.plot(
        epochs,
        values,
        color=color,
        linestyle=linestyle,
        linewidth=2.25,
        label=label,
    )
    ax.scatter(
        epochs[-1],
        values[-1],
        color=color,
        marker=marker,
        s=46,
        zorder=3,
    )
    return line, epochs[-1], values[-1]


def annotate_final(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    color: str,
    y_offset: float = 0,
) -> None:
    ax.annotate(
        f"{y:+.2f}",
        (x, y),
        xytext=(7, y_offset),
        textcoords="offset points",
        ha="left",
        va="center",
        color=color,
        fontsize=9.5,
        fontweight="bold",
    )


def plot_phase1_validation_curves() -> None:
    runs = [
        (
            "A · 124 uniform",
            "growth24_projected_a_v1_mdp_train",
            BLUE,
            "-",
            "o",
        ),
        (
            "B · 96/144/144/240",
            "growth24_projected_b_v1_mdp_train",
            ORANGE,
            "--",
            "s",
        ),
        (
            "C · 96/192/336/456",
            "growth24_projected_c_v1_mdp_train",
            GREEN,
            "-.",
            "^",
        ),
    ]
    histories = [
        (label, read_history(run), color, linestyle, marker)
        for label, run, color, linestyle, marker in runs
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)
    endpoint_offsets = {
        "type1_median": [10, 0, -11],
        "type2_median": [-8, 0, 9],
    }
    handles: list[Line2D] = []
    for ax, key, panel_title in [
        (axes[0], "type1_median", "Unseen layout · type1"),
        (axes[1], "type2_median", "Seen layout · type2"),
    ]:
        for index, (label, history, color, linestyle, marker) in enumerate(histories):
            line, x_final, y_final = add_validation_curve(
                ax,
                history,
                key=key,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
            )
            if key == "type1_median":
                handles.append(line)
            annotate_final(
                ax,
                x_final,
                y_final,
                color=color,
                y_offset=endpoint_offsets[key][index],
            )
        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=9)
        ax.set_xlabel("Epoch")
        style_validation_axis(ax, y_limits=(0, 8.05), x_max=1110)

    fig.suptitle(
        "Phase I · g24/MDP validation dynamics",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.supylabel("Median WaveSDR (dB)", x=0.012)
    fig.legend(
        handles=handles,
        labels=[run[0] for run in runs],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.927),
        ncol=3,
        handlelength=2.8,
    )
    fig.text(
        0.5,
        0.012,
        "EnsembleSet validation · warmup-cosine · 6 works / 48 queries · every 25 epochs · seed 2026",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.025, 0.065, 1, 0.895))
    save_figure(fig, "phase1_type1_vs_type2")


def plot_phase2_validation_swap() -> None:
    curves = [
        (
            "No swap",
            read_history("SplitStagePaper_1000ep_warmup_cosine"),
            BLUE,
            "-",
            "o",
        ),
        (
            "Swap p=0.15",
            read_history("SSP_dense124_swap15"),
            ORANGE,
            "--",
            "s",
        ),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for label, history, color, linestyle, marker in curves:
        _, x_final, y_final = add_validation_curve(
            ax,
            history,
            key="type1_median",
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )
        annotate_final(ax, x_final, y_final, color=color)

    ax.set_title(
        "Phase II · dense124/SSP validation response to position swap",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Type1 median WaveSDR (dB)")
    style_validation_axis(ax, y_limits=(-0.35, 6.75), x_max=1065)
    ax.legend(frameon=False, loc="center right")
    fig.text(
        0.5,
        0.012,
        "EnsembleSet validation · warmup-cosine · 6 works / 48 queries · every 25 epochs · seed 2026",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, "phase2_swap_effect")


def plot_phase3_validation_routes() -> None:
    curves = [
        (
            "MDP · no swap",
            read_history("growth24_projected_b_v1_mdp_train"),
            BLUE,
            "-",
            "o",
            -8,
        ),
        (
            "MDP · swap p=0.15",
            read_history("growth24_projected_b_v1_mdp_swap15"),
            BLUE,
            "--",
            "s",
            13,
        ),
        (
            "SSP · no swap",
            read_history("growth24_projected_b_v1_train"),
            ORANGE,
            "-",
            "o",
            0,
        ),
        (
            "SSP · swap p=0.15",
            read_history("growth24_projected_b_v1_swap15"),
            ORANGE,
            "--",
            "s",
            -18,
        ),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for label, history, color, linestyle, marker, y_offset in curves:
        _, x_final, y_final = add_validation_curve(
            ax,
            history,
            key="type1_median",
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )
        annotate_final(
            ax,
            x_final,
            y_final,
            color=color,
            y_offset=y_offset,
        )

    ax.set_title(
        "Phase III · position swap transfers across g24 routes",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Type1 median WaveSDR (dB)")
    style_validation_axis(ax, y_limits=(-0.25, 6.15), x_max=1065)
    ax.legend(frameon=False, loc="center right", ncol=2)
    fig.text(
        0.5,
        0.012,
        "EnsembleSet validation · warmup-cosine · 6 works / 48 queries · every 25 epochs · seed 2026",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, "phase3_swap_transfer")


def plot_phase3_validation_lr_swap() -> None:
    curves = [
        (
            "Warmup-cosine · no swap",
            read_history("growth24_projected_b_v1_mdp_train"),
            BLUE,
            "-",
            "o",
            -8,
        ),
        (
            "Warmup-cosine · swap p=0.15",
            read_history("growth24_projected_b_v1_mdp_swap15"),
            BLUE,
            "--",
            "s",
            14,
        ),
        (
            "Constant · no swap",
            read_history("growth24_projected_b_v1_mdp_constlr_train"),
            ORANGE,
            "-",
            "o",
            4,
        ),
        (
            "Constant · swap p=0.15",
            read_history("growth24_projected_b_v1_mdp_constlr_swap15"),
            ORANGE,
            "--",
            "s",
            -14,
        ),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for label, history, color, linestyle, marker, y_offset in curves:
        _, x_final, y_final = add_validation_curve(
            ax,
            history,
            key="type1_median",
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )
        annotate_final(
            ax,
            x_final,
            y_final,
            color=color,
            y_offset=y_offset,
        )

    ax.set_title(
        "Phase III · g24/MDP learning-rate × swap interaction",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Type1 median WaveSDR (dB)")
    style_validation_axis(ax, y_limits=(-0.25, 6.15), x_max=1065)
    ax.legend(frameon=False, loc="center right", ncol=2)
    fig.text(
        0.5,
        0.012,
        "EnsembleSet validation · 6 works / 48 queries · every 25 epochs · seed 2026",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, "phase3_lr_swap_interaction")


def plot_phase2_dose_response() -> None:
    runs = [
        (0.00, "TruePure_1000ep_warmup_cosine/test_results.json"),
        (0.10, "PureSwap_E1cosine-P10_1000ep_p010/test_results.json"),
        (0.15, "PureSwap_E1cosine-P15_1000ep_p015/test_results.json"),
        (
            0.25,
            "historical_test_results/"
            "PureSwap_E1cosine-P25_1000ep_p025_test_results.json",
        ),
        (0.50, "PureSwap_E1cosine-P50_1000ep_p050/test_results.json"),
    ]
    probabilities = [probability for probability, _ in runs]
    final_rows = [read_final(path) for _, path in runs]
    type1 = [row["type1_median"] for row in final_rows]
    type2 = [row["type2_median"] for row in final_rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    ax.plot(
        probabilities,
        type1,
        color=ORANGE,
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Unseen layout · type1",
    )
    ax.plot(
        probabilities,
        type2,
        color=BLUE,
        marker="s",
        markersize=7,
        linewidth=2.4,
        linestyle="--",
        label="Seen layout · type2",
    )

    for x, y in zip(probabilities, type1):
        ax.annotate(
            f"{y:+.2f}",
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=ORANGE,
            fontsize=10,
            fontweight="bold",
        )
    for x, y in zip(probabilities, type2):
        ax.annotate(
            f"{y:+.2f}",
            (x, y),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            color=BLUE,
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(
        "Phase II · dense124/MDP real-test swap dose response",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Position-swap probability p")
    ax.set_ylabel("Median WaveSDR (dB)")
    ax.set_xticks(probabilities, ["0", "0.10", "0.15", "0.25", "0.50"])
    ax.set_xlim(-0.025, 0.525)
    ax.set_ylim(-0.55, 2.65)
    style_numeric_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.text(
        0.5,
        0.012,
        "URMP real test · warmup-cosine · 6 works / 48 queries · seed 2026",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, "phase2_mdp_swap_dose_response")


def plot_phase3_paired_test() -> None:
    pairs = [
        (
            "g24 B · MDP · wcos",
            "growth24_projected_b_v1_mdp_train/test_results.json",
            "growth24_projected_b_v1_mdp_swap15/test_results.json",
        ),
        (
            "g24 B · SSP · wcos",
            "growth24_projected_b_v1_train/test_results.json",
            "growth24_projected_b_v1_swap15/test_results.json",
        ),
        (
            "g24 B · MDP · const",
            "growth24_projected_b_v1_mdp_constlr_train/test_results.json",
            "growth24_projected_b_v1_mdp_constlr_swap15/test_results.json",
        ),
        (
            "g24 B · SSP · const",
            "growth24_projected_b_v1_constlr_train/test_results.json",
            "growth24_projected_b_v1_constlr_swap15/test_results.json",
        ),
        (
            "dense124 · MDP · wcos",
            "TruePure_1000ep_warmup_cosine/test_results.json",
            "PureSwap_E1cosine-P15_1000ep_p015/test_results.json",
        ),
        (
            "dense124 · SSP · wcos",
            "SplitStagePaper_1000ep_warmup_cosine/test_results.json",
            "SSP_dense124_swap15/test_results.json",
        ),
    ]
    labels = [label for label, _, _ in pairs]
    no_swap = [read_final(base)["type1_median"] for _, base, _ in pairs]
    swap = [read_final(intervention)["type1_median"] for _, _, intervention in pairs]
    y_values = list(range(len(pairs) - 1, -1, -1))

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    for y, base_value, swap_value in zip(y_values, no_swap, swap):
        ax.plot(
            [base_value, swap_value],
            [y, y],
            color="#AAB1B7",
            linewidth=2.0,
            zorder=1,
        )
        ax.scatter(base_value, y, color=BLUE, marker="o", s=75, zorder=3)
        ax.scatter(swap_value, y, color=ORANGE, marker="s", s=75, zorder=3)
        ax.annotate(
            f"{base_value:+.2f}",
            (base_value, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=BLUE,
            fontsize=9.5,
            fontweight="bold",
        )
        ax.annotate(
            f"{swap_value:+.2f}",
            (swap_value, y),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            color=ORANGE,
            fontsize=9.5,
            fontweight="bold",
        )
        ax.text(
            2.47,
            y,
            f"Δ {swap_value - base_value:+.2f}",
            ha="left",
            va="center",
            color=TEXT,
            fontsize=10,
        )

    ax.set_title(
        "Phase III · position swap improves real-test unseen-layout separation",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("URMP test type1 median WaveSDR (dB)")
    ax.set_yticks(y_values, labels)
    ax.set_xlim(-0.5, 2.85)
    ax.set_ylim(-0.7, len(pairs) - 0.3)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.axvline(0, color="#9BA3AA", linewidth=0.9)
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", linestyle="None", label="No swap"),
            Line2D(
                [],
                [],
                color=ORANGE,
                marker="s",
                linestyle="None",
                label="Swap p=0.15",
            ),
        ],
        frameon=False,
        loc="lower right",
    )
    fig.text(
        0.5,
        0.012,
        "URMP real test · 6 works / 48 queries per run · seed 2026 · Δ uses unrounded medians",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, "phase3_test_type1_paired")


def add_stage_box(
    ax: plt.Axes,
    x: float,
    title: str,
    domain: str,
    detail: str,
    counts: str,
    color: str,
) -> None:
    box = FancyBboxPatch(
        (x, 0.34),
        0.255,
        0.42,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(box)
    ax.text(x + 0.1275, 0.69, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(x + 0.1275, 0.60, domain, ha="center", va="center", fontsize=12, color=color, fontweight="bold")
    ax.text(x + 0.1275, 0.50, detail, ha="center", va="center", fontsize=11.5)
    ax.text(x + 0.1275, 0.40, counts, ha="center", va="center", fontsize=10.5, color=MUTED)


def plot_study_design() -> None:
    train = read_manifest("sqr_paper_split_train.jsonl")
    validation = read_manifest("sqr_paper_split_val.jsonl")
    test = read_manifest("sqr_paper_split_test.jsonl")
    train_works = len({row["composition_id"] for row in train})
    validation_works = len({row["composition_id"] for row in validation})
    test_works = len({row["composition_id"] for row in test})

    if (len(train), len(validation), len(test)) != (59, 12, 12):
        raise ValueError("Unexpected manifest sizes; update the study-design figure")

    fig, ax = plt.subplots(figsize=(13.5, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.82)
    ax.axis("off")
    ax.set_title(
        "Study design · synthetic-to-real and seen-to-unseen layout shifts",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )

    add_stage_box(
        ax,
        0.035,
        "TRAIN",
        "Synthetic · EnsembleSet",
        "type2 only · seen layout",
        f"{train_works} works · {len(train)} records",
        BLUE,
    )
    add_stage_box(
        ax,
        0.3725,
        "VALIDATION",
        "Synthetic · EnsembleSet",
        "type1 + type2",
        f"{validation_works} works · {len(validation)} records · 48 queries",
        BLUE,
    )
    add_stage_box(
        ax,
        0.71,
        "FINAL TEST",
        "Real · URMP",
        "type1 + type2",
        f"{test_works} works · {len(test)} records · 48 queries",
        ORANGE,
    )

    for start, end, label in [
        ((0.293, 0.55), (0.365, 0.55), "every 25 epochs"),
        ((0.6305, 0.55), (0.7025, 0.55), "final only"),
    ]:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#7D858D",
        )
        ax.add_patch(arrow)
        ax.text(
            (start[0] + end[0]) / 2,
            0.59,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=MUTED,
        )

    domain_band = FancyBboxPatch(
        (0.10, 0.16),
        0.80,
        0.08,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0,
        facecolor=ORANGE + "16",
    )
    ax.add_patch(domain_band)
    ax.text(0.16, 0.20, "DOMAIN SHIFT", ha="left", va="center", fontsize=10.5, color=ORANGE, fontweight="bold")
    ax.text(0.40, 0.20, "sampled / rendered performances", ha="center", va="center", fontsize=10.5)
    ax.text(0.605, 0.20, "→", ha="center", va="center", fontsize=15, color=ORANGE)
    ax.text(0.76, 0.20, "human performances", ha="center", va="center", fontsize=10.5)

    layout_band = FancyBboxPatch(
        (0.10, 0.045),
        0.80,
        0.08,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0,
        facecolor=PURPLE + "16",
    )
    ax.add_patch(layout_band)
    ax.text(0.16, 0.085, "LAYOUT SHIFT", ha="left", va="center", fontsize=10.5, color=PURPLE, fontweight="bold")
    ax.text(0.40, 0.085, "training sees type2 only", ha="center", va="center", fontsize=10.5)
    ax.text(0.605, 0.085, "→", ha="center", va="center", fontsize=15, color=PURPLE)
    ax.text(0.76, 0.085, "type1 is unseen", ha="center", va="center", fontsize=10.5)

    fig.tight_layout()
    save_figure(fig, "study_design")


def main() -> None:
    plot_phase1_validation_curves()
    plot_phase2_validation_swap()
    plot_phase3_validation_routes()
    plot_phase3_validation_lr_swap()
    plot_phase2_dose_response()
    plot_phase3_paired_test()
    plot_study_design()


if __name__ == "__main__":
    main()
