from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

HSE_DARK_BLUE = "#003A70"
HSE_LIGHT_BLUE = "#5B9BD5"
HSE_GREY = "#7A7F85"
HSE_LIGHT_GREY = "#D9DEE3"
HSE_TEXT = "#222222"

MODEL_DISPLAY_ORDER = ["llama3", "qwen", "yandex"]
MODEL_PROMPT_ORDER = ["yandex", "llama3", "qwen"]
MODEL_LABELS = {"yandex": "YandexGPT", "llama3": "Llama 3", "qwen": "Qwen"}
PROMPT_ORDER = ["baseline", "provoking", "conservative"]
PROMPT_COLORS = {"baseline": HSE_DARK_BLUE, "provoking": HSE_LIGHT_BLUE, "conservative": HSE_GREY}


def setup_plot():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": HSE_LIGHT_GREY,
        "axes.labelcolor": HSE_TEXT,
        "xtick.color": HSE_TEXT,
        "ytick.color": HSE_TEXT,
        "text.color": HSE_TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def style_axis(ax):
    ax.grid(False)
    ax.spines["left"].set_color(HSE_LIGHT_GREY)
    ax.spines["bottom"].set_color(HSE_LIGHT_GREY)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", length=3.5, width=0.8, color=HSE_TEXT)


def add_value_labels(ax, bars, values, dy=1.2):
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=HSE_TEXT,
        )


def save_figure(fig, filename):
    out = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def read_metrics():
    by_model = pd.read_csv(METRICS_DIR / "metrics_by_model.csv")
    by_prompt = pd.read_csv(METRICS_DIR / "metrics_by_model_prompt.csv")
    return by_model, by_prompt


def ordered_by_model(df, order=MODEL_DISPLAY_ORDER):
    return df.set_index("model").loc[order].reset_index()


def plot_svr_fr_by_model(by_model):
    df = ordered_by_model(by_model)
    labels = [MODEL_LABELS[m] for m in df["model"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars1 = ax.bar(x - width / 2, df["svr_rate"] * 100, width, label="SVR", color=HSE_DARK_BLUE, linewidth=0)
    bars2 = ax.bar(x + width / 2, df["fr_rate"] * 100, width, label="FR", color=HSE_LIGHT_BLUE, linewidth=0)
    add_value_labels(ax, bars1, df["svr_rate"] * 100)
    add_value_labels(ax, bars2, df["fr_rate"] * 100)

    ax.set_ylabel("Доля наблюдений, %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, handlelength=1.8, columnspacing=1.5)
    save_figure(fig, "fig_3_1_svr_fr_by_model.png")


def plot_applicability_funnel(by_model):
    total = int(by_model["n_responses"].sum())
    stages = [
        ("Все ответы", int(by_model["n_responses"].sum())),
        ("Заявлен источник", int(by_model["source_metric_n"].sum())),
        ("Источник идентифицирован", int(by_model["source_identified_n"].sum())),
        ("Цитата проверяема", int(by_model["quote_checkable_n"].sum())),
        ("Цитата подтверждена", int(by_model["quote_verified_n"].sum())),
        ("Тезис поддержан", int(by_model["claim_supported_n"].sum())),
        ("LVR применим", int(by_model["locator_checkable_n"].sum())),
    ]

    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    percents = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y = np.arange(len(labels))
    colors = [HSE_DARK_BLUE, HSE_DARK_BLUE, HSE_LIGHT_BLUE, HSE_LIGHT_BLUE, HSE_GREY, HSE_GREY, HSE_GREY]
    bars = ax.barh(y, percents, color=colors, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Доля от всех ответов, %")
    ax.set_xlim(0, 105)
    style_axis(ax)
    for bar, count, pct in zip(bars, counts, percents):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{count} ({pct:.1f}%)", va="center", fontsize=8.5, color=HSE_TEXT)
    save_figure(fig, "fig_3_2_applicability_funnel.png")


def plot_mr_qvr_esr_by_model(by_model):
    df = ordered_by_model(by_model)
    labels = [MODEL_LABELS[m] for m in df["model"]]
    x = np.arange(len(labels))
    width = 0.23
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for j, (col, label, color) in enumerate([("mr_rate", "MR", HSE_DARK_BLUE), ("qvr_rate", "QVR", HSE_LIGHT_BLUE), ("esr_rate", "ESR", HSE_GREY)]):
        values = df[col] * 100
        bars = ax.bar(x + (j - 1) * width, values, width, label=label, color=color, linewidth=0)
        add_value_labels(ax, bars, values)
    ax.set_ylabel("Доля наблюдений, %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, handlelength=1.8, columnspacing=1.5)
    save_figure(fig, "fig_3_3_mr_qvr_esr_by_model.png")


def plot_quote_locator_coverage(by_model):
    df = ordered_by_model(by_model)
    labels = [MODEL_LABELS[m] for m in df["model"]]
    x = np.arange(len(labels))
    width = 0.24
    quote_field = df["quote_text_nonempty_n"] / df["n_responses"] * 100
    locator_field = df["source_locator_nonempty_n"] / df["n_responses"] * 100
    quote_verified = df["quote_verified_n"] / df["n_responses"] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, (values, label, color) in enumerate([
        (quote_field, "Поле quote заполнено", HSE_DARK_BLUE),
        (locator_field, "Локатор указан", HSE_LIGHT_BLUE),
        (quote_verified, "Цитата подтверждена", HSE_GREY),
    ]):
        bars = ax.bar(x + (i - 1) * width, values, width, label=label, color=color, linewidth=0)
        add_value_labels(ax, bars, values)
    ax.set_ylabel("Доля от всех ответов, %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, handlelength=1.8, columnspacing=1.2)
    save_figure(fig, "fig_3_4_quote_locator_coverage.png")


def plot_prompt_metric(by_prompt, metric, ylabel, filename):
    pivot = by_prompt.pivot(index="model", columns="prompt_type", values=metric).reindex(MODEL_PROMPT_ORDER).reindex(columns=PROMPT_ORDER)
    model_labels = [MODEL_LABELS[m] for m in MODEL_PROMPT_ORDER]
    x = np.arange(len(MODEL_PROMPT_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, prompt in enumerate(PROMPT_ORDER):
        values = pivot[prompt] * 100
        bars = ax.bar(x + (i - 1) * width, values, width, label=prompt, color=PROMPT_COLORS[prompt], linewidth=0)
        add_value_labels(ax, bars, values)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, len(MODEL_PROMPT_ORDER) - 0.5)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, handlelength=1.8, columnspacing=1.5)
    save_figure(fig, filename)


def make_all_figures():
    setup_plot()
    by_model, by_prompt = read_metrics()
    plot_svr_fr_by_model(by_model)
    plot_applicability_funnel(by_model)
    plot_mr_qvr_esr_by_model(by_model)
    plot_quote_locator_coverage(by_model)
    plot_prompt_metric(by_prompt, "svr_rate", "Доля идентифицируемых источников, %", "fig_3_5_svr_by_prompt.png")
    plot_prompt_metric(by_prompt, "fr_rate", "Доля сфабрикованных источников, %", "fig_3_6_fr_by_prompt.png")


if __name__ == "__main__":
    make_all_figures()
