#!/usr/bin/env python3
from pydantic import BaseModel
import openai
from typing import List, Dict, Literal
import os
from Scripts.LLM import call_GPT
from dotenv import load_dotenv
import logging
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures

# ----------------------------------------------------------------------------#
# ---- Define Response Structure ---- #
# ----------------------------------------------------------------------------#

class Response_LLM(BaseModel):
    overall_relevance: int
    binary_response: Literal["relevant", "irrelevant"]

# ----------------------------------------------------------------------------#
# ---- Helper functions for group extraction ---- #
# ----------------------------------------------------------------------------#

def extract_barrier_group(barrier_text: str):
    match = re.search(r"inside barrier group '([^']+)'", barrier_text)
    return match.group(1) if match else None


def extract_coping_group(coping_text: str):
    match = re.search(r"inside coping option group '([^']+)'", coping_text)
    return match.group(1) if match else None


# ----------------------------------------------------------------------------#
# ---- Confusion‑matrix helpers ---- #
# ----------------------------------------------------------------------------#

def _confusion_df(true_labels: list[str], pred_labels: list[str]) -> pd.DataFrame:
    """Return a 2 × 2 confusion‑matrix DataFrame in fixed order."""
    idx = pd.Index(["relevant", "irrelevant"], name="Actual")
    cols = pd.Index(["relevant", "irrelevant"], name="Predicted")
    ct = pd.crosstab(pd.Series(true_labels), pd.Series(pred_labels))
    # re‑index so missing rows / cols become zeros
    return ct.reindex(index=idx, columns=cols, fill_value=0)


def _plot_confusion(cm_df: pd.DataFrame, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_xticks(range(2))
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticks(range(2))
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, pad=15)
    for (i, j), val in np.ndenumerate(cm_df.values):
        ax.text(j, i, str(val), ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------#
# ---- Core pipeline functions ---- #
# ----------------------------------------------------------------------------#

def create_combinations(file_path_1: str, verbose_mode: bool = False) -> pd.DataFrame:
    logging.info("Creating combinations of 'barriers' and 'coping options'...")
    df = pd.read_csv(file_path_1)

    barriers, coping_options, user_ratings = [], [], []

    for _, row in df.iterrows():
        barrier_combination = (
            f"Barrier label '{row['label_barrier']}' with barrier instance: '{row['Instance_barrier']}' ; "
            f"inside barrier group '{row['barrier_group']}'"
        )
        barriers.append(barrier_combination)

        coping_combination = (
            f"Coping option label '{row['label_solution']}' with coping option instance: '{row['Instance_solution']}' ; "
            f"inside coping option group '{row['Category']}'"
        )
        coping_options.append(coping_combination)

        user_ratings.append(
            [row["Rel_Always"], row["Rel_CC"], row["Rel_Never"], row["Relevance"]]
        )

    combinations_df = pd.DataFrame(
        {
            "Barrier": barriers,
            "Coping Option": coping_options,
            "Aggregated User Rating": user_ratings,
            "Original Barrier": df["barrier_group"],
            "Original Coping Group": df["Category"],
        }
    )

    if verbose_mode:
        logging.info(f"Created {len(combinations_df)} barrier–coping combinations")

    return combinations_df


def generate_LLM_ratings(
    combinations_df: pd.DataFrame,
    max_workers: int = 5,
    LLM: str = "gpt-4o-mini",
    verbose_mode: bool = False,
    test_mode: bool = False,
) -> List[Response_LLM]:
    logging.info("Generating LLM scores for combinations using ThreadPoolExecutor...")

    system_prompt = ("You are an expert in the field of interdisciplinary physical activity research. "
                     "You are tasked to evaluate the relevance of a certain 'coping option' for a specific 'barrier' ; The goal is to find appropriate coping options for barriers to ultimately increase daily physical activity levels. "
                     "You will provide an integer that represents the overall relevance of the coping option for the barrier ; this is from 0 to 1000 (0=completely irrelevant, 1000=perfectly relevant). "
                     "Finally, you will provide a binary response that summarizes whether the coping option can be considered relevant or not, for that specific barrier. "
                     "; keep in mind that 73.30% of all combinations should be labeled as relevant ; 26.70% should not! Pay attention for the irrelevance during scoring. "
                     "IMPORTANT: For the overall relevance score, ensure to be extremely detailled, therefore use increments of 1."
                     "Avoid classifying coping options as 'irrelevant, when actually they are relevant (i.e., false negatives), "
                     )

    responses: dict[int, concurrent.futures.Future] = {}

    def wrapped_call(idx: int, user_query: str):
        result: Response_LLM = call_GPT(
            system_prompt=system_prompt,
            user_query=user_query,
            pydantic_model=Response_LLM,
            model=LLM,
        )
        logging.info(f"Processed item {idx + 1}/{len(combinations_df)}")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in combinations_df.iterrows():
            if test_mode and idx >= 500:
                break
            query = (
                f"Evaluate the relevance of the COPING OPTION '{row['Coping Option']}' "
                f"for the BARRIER '{row['Barrier']}'."
            )
            responses[idx] = executor.submit(wrapped_call, idx, query)

    return [responses[i].result() for i in sorted(responses.keys())]


# ----------------------------------------------------------------------------#
# ---- Metrics & visualisations ---- #
# ----------------------------------------------------------------------------#

import pandas as pd
import itertools
import logging
from typing import List

def compute_correlation_and_confusion(
    combinations_df: pd.DataFrame,
    llm_scores: List[Response_LLM],
    corr_path: str,
    conf_path: str,
) -> pd.DataFrame:
    """
    Build a row‑level data frame, min‑max scale all numeric variables (0‑1),
    compute a Pearson correlation matrix, show raw pairwise scatter plots
    (with red regression lines), and save the correlation‑heat‑map and
    confusion‑matrix figures.

    Parameters
    ----------
    combinations_df : pd.DataFrame
        Data frame containing user ratings, barriers, and coping options.
    llm_scores : List[Response_LLM]
        List of LLM responses with overall relevance scores and binary decisions.
    corr_path : str
        File path to save the correlation‑matrix heat map (PNG recommended).
    conf_path : str
        File path to save the confusion‑matrix plot (PNG recommended).

    Returns
    -------
    pd.DataFrame
        The per‑example frame (scaled) used to build the correlation matrix.
    """
    # ------------------------------------------------------------------ #
    # Build the per‑example data frame                                   #
    # ------------------------------------------------------------------ #
    corr_rows, true_labels, pred_labels = [], [], []
    for i, resp in enumerate(llm_scores):
        user_vals = combinations_df.loc[i, "Aggregated User Rating"]

        corr_rows.append(
            {
                "always_USER": float(user_vals[0]),
                "never_USER": float(user_vals[2]),
                "overall_LLM": resp.overall_relevance,
            }
        )
        true_labels.append(str(user_vals[3]).strip().lower())
        pred_labels.append(str(resp.binary_response).strip().lower())

    corr_df = pd.DataFrame(corr_rows)
    corr_df["barrier_group"] = combinations_df["Barrier"].apply(extract_barrier_group)
    corr_df["coping_group"] = combinations_df["Coping Option"].apply(
        extract_coping_group
    )

    # ------------------------------------------------------------------ #
    # Min‑max scaling (0‑1) for numeric columns                          #
    # ------------------------------------------------------------------ #
    numeric_cols = ["always_USER", "never_USER", "overall_LLM"]
    for col in numeric_cols:
        col_min, col_max = corr_df[col].min(), corr_df[col].max()
        if col_max > col_min:
            corr_df[col] = (corr_df[col] - col_min) / (col_max - col_min)
        else:  # constant column
            corr_df[col] = 0.0

    # ------------------------------------------------------------------ #
    # 1) Pearson correlation heat map – saved to ``corr_path``           #
    # ------------------------------------------------------------------ #
    corr_mat = corr_df[numeric_cols].corr("pearson")

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(corr_mat, cmap="viridis")
    ax.set_xticks(range(len(corr_mat)))
    ax.set_xticklabels(corr_mat.columns, rotation=45, ha="left")
    ax.set_yticks(range(len(corr_mat)))
    ax.set_yticklabels(corr_mat.index)
    fig.colorbar(cax, ax=ax)
    ax.set_title("Correlation Matrix: full dataset", pad=20, weight="bold", fontsize=16)

    for (i, j), val in np.ndenumerate(corr_mat.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")

    fig.tight_layout()
    fig.savefig(corr_path, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2) Raw scatter plots with red regression lines (display only)      #
    # ------------------------------------------------------------------ #
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (x_col, y_col) in enumerate(itertools.combinations(numeric_cols, 2)):
        fig, ax = plt.subplots(figsize=(6, 5))

        x = corr_df[x_col].values
        y = corr_df[y_col].values

        scatter_color = color_cycle[idx % len(color_cycle)]

        # Raw points
        ax.scatter(x, y, alpha=0.75, s=40, label="raw data", color=scatter_color)

        # Least‑squares fit
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit,
            y_fit,
            linewidth=2.5,
            label=f"fit: y = {slope:.2f}x + {intercept:.2f}",
            color="red",
        )

        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.set_title(f"{y_col} vs. {x_col}", weight="bold", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # 3) Confusion matrix (saved)                                        #
    # ------------------------------------------------------------------ #
    cm_df = _confusion_df(true_labels, pred_labels)
    _plot_confusion(cm_df, "Global confusion matrix", conf_path)

    acc = (
        cm_df.loc["relevant", "relevant"] + cm_df.loc["irrelevant", "irrelevant"]
    ) / len(true_labels)
    logging.info(f"Global classification accuracy = {acc:.2%}")

    return corr_df




def grouped_metrics(
    corr_df: pd.DataFrame,
    combinations_df: pd.DataFrame,
    llm_scores: List[Response_LLM],
    base_dir: str,
) -> None:
    """Correlation & confusion matrices for every barrier & coping group."""
    barrier_dir = os.path.join(base_dir, "separate", "barrier_groups")
    coping_dir = os.path.join(base_dir, "separate", "coping_option_groups")
    os.makedirs(barrier_dir, exist_ok=True)
    os.makedirs(coping_dir, exist_ok=True)

    def _group_loop(level: str, grp_dir: str, col_name: str) -> None:
        for grp in corr_df[col_name].dropna().unique():
            idxs = corr_df[corr_df[col_name] == grp].index
            if len(idxs) < 2:
                continue

            # correlation
            sub_corr = corr_df.loc[
                idxs, ["always_USER", "never_USER", "overall_LLM"]
            ].corr("pearson")
            fig, ax = plt.subplots(figsize=(8, 7))
            cax = ax.matshow(sub_corr)
            ax.set_xticks(range(len(sub_corr)))
            ax.set_xticklabels(sub_corr.columns, rotation=45, ha="left")
            ax.set_yticks(range(len(sub_corr)))
            ax.set_yticklabels(sub_corr.index)
            fig.colorbar(cax, ax=ax)
            ax.set_title(f"Correlation Matrix: {level} '{grp}'", pad=20)
            for (i, j), val in np.ndenumerate(sub_corr.values):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")
            fig.tight_layout()
            fig.savefig(
                os.path.join(grp_dir, f"correlation_matrix_{level}_{grp}.png"), dpi=300
            )
            plt.close(fig)

            # confusion
            true_labels = [
                str(combinations_df.loc[i, "Aggregated User Rating"][3]).strip().lower()
                for i in idxs
            ]
            pred_labels = [
                str(llm_scores[i].binary_response).strip().lower() for i in idxs
            ]
            cm_df = _confusion_df(true_labels, pred_labels)
            acc = (
                cm_df.loc["relevant", "relevant"]
                + cm_df.loc["irrelevant", "irrelevant"]
            ) / len(idxs)
            _plot_confusion(
                cm_df,
                f"{level} '{grp}' confusion matrix (acc={acc:.2%})",
                os.path.join(grp_dir, f"confusion_matrix_{level}_{grp}.png"),
            )
            logging.info(f"{level} '{grp}' accuracy = {acc:.2%}")

    _group_loop("barrier_group", barrier_dir, "barrier_group")
    _group_loop("coping_group", coping_dir, "coping_group")


def save_llm_results_excel(
    combinations_df: pd.DataFrame,
    llm_scores: List[Response_LLM],
    output_file: str,
) -> None:
    records = []
    for i, resp in enumerate(llm_scores):
        records.append(
            {
                "Barrier": combinations_df.loc[i, "Barrier"],
                "Coping Option": combinations_df.loc[i, "Coping Option"],
                "LLM Result": f"Overall: {resp.overall_relevance}, "
                f"Binary: {resp.binary_response}",
            }
        )
    (pd.DataFrame(records).pivot(index="Barrier", columns="Coping Option", values="LLM Result")
        .to_excel(output_file))
    logging.info(f"Raw LLM results written to {output_file}")


# ----------------------------------------------------------------------------#
# ---- Main entry point ---- #
# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY") or ""

    # ----- 1. Create combinations -------------------------------------------------
    file_path = "/OSF_data/relevance/relevance_by_combination.csv"
    combinations_df = create_combinations(file_path, verbose_mode=False)

    # ----- 2. LLM evaluation ------------------------------------------------------
    llm_scores = generate_LLM_ratings(
        combinations_df,
        max_workers=100,
        LLM="gpt-4o-mini",
        verbose_mode=True,
        test_mode=False,
    )

    # ----- 3. Metrics + visualisations -------------------------------------------
    base_image_dir = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/Ondernemingen/PECAN2.0/Documenten/Comparison_UserRatings_LLMRatings/results/images"
    full_dir = os.path.join(base_image_dir, "full")
    os.makedirs(full_dir, exist_ok=True)

    corr_df = compute_correlation_and_confusion(
        combinations_df,
        llm_scores,
        corr_path=os.path.join(full_dir, "correlation_matrix.png"),
        conf_path=os.path.join(full_dir, "confusion_matrix.png"),
    )
    grouped_metrics(corr_df, combinations_df, llm_scores, base_image_dir)

    logging.info("Done – all correlation & confusion matrices saved.")
