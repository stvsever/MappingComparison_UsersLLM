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
# ---- Option 1 ---- # ; without conditions
# ----------------------------------------------------------------------------#

class Response_LLM_nuanced(BaseModel):
    always_relevant: float
    conditionally_relevant: float
    never_relevant: float


class Response_LLM(BaseModel):
    nuanced_response: Response_LLM_nuanced
    overall_relevance: float
    binary_response: Literal["relevant", "irrelevant"]


# ----------------------------------------------------------------------------#
# ---- Option 2 ---- # ; with conditions
# ----------------------------------------------------------------------------#

# class Response_LLM_conditions(BaseModel):
#    materials_required: float
#    materials_not_required: float
#
#    activity_planned_alone: float
#    activity_planned_with_other_people: float
#
#    activity_planned_indoors: float
#    activity_planned_outside: float
#
#    activity_is_high_intensity: float
#    activity_is_low_intensity: float
#
#
# class Response_LLM_conditional(BaseModel):
#    conditionally_relevant: float
#    conditions: Response_LLM_conditions
#
#
# class Response_LLM_nuanced(BaseModel): # ; alternatively: can additionaly include user-based clusters
#    always_relevant: float
#    conditionally_relevant: Response_LLM_conditional
#    never_relevant: float
#
#
# class Response_LLM(BaseModel):  # expected number of LLM_responses: 50 (barriers) x 64 (coping options) = 3200
#    nuanced_response: Response_LLM_nuanced
#    binary_response: Literal["relevant", "irrelevant"]

# ----------------------------------------------------------------------------#
# ---- Helper functions for group extraction ---- #
# ----------------------------------------------------------------------------#

def extract_barrier_group(barrier_text):
    # Extract barrier group from the barrier combination string
    match = re.search(r"inside barrier group '([^']+)'", barrier_text)
    if match:
        return match.group(1)
    else:
        return None

def extract_coping_group(coping_text):
    # Extract coping option group from the coping combination string
    match = re.search(r"inside coping option group '([^']+)'", coping_text)
    if match:
        return match.group(1)
    else:
        return None

# ----------------------------------------------------------------------------#
# ---- Functions ---- #
# ----------------------------------------------------------------------------#

def create_combinations(file_path_1, verbose_mode=False):
    logging.info("Creating combinations of 'barriers' and 'coping options'...")

    # Read the CSV file
    df = pd.read_csv(file_path_1)

    # Initialize empty lists
    barriers = []
    coping_options = []
    user_ratings = []

    # Extract barriers and coping options
    for index, row in df.iterrows():
        # 1. Extract barriers (has label, instance & group) ; stored as 'label_barrier', 'Instance_barrier', and 'barrier_group'
        barrier_label = row["label_barrier"]
        barrier_instance = row["Instance_barrier"]
        barrier_group = row["barrier_group"]

        # Combine them into a single string
        barrier_combination = f"Barrier label '{barrier_label}' with barrier instance: '{barrier_instance}' ; inside barrier group '{barrier_group}'"
        barriers.append(barrier_combination)

        # 2. Extract coping options (has label, instance & group) ; stored as 'label_solution', 'Instance_solution', and 'Category'
        coping_label = row["label_solution"]
        coping_instance = row["Instance_solution"]
        coping_group = row["Category"]

        # Combine them into a single string
        coping_combination = f"Coping option label '{coping_label}' with coping option instance: '{coping_instance}' ; inside coping option group '{coping_group}'"
        coping_options.append(coping_combination)

        # 3. Extract 4 other measures of interest (stored as 'Rel_Always', 'Rel_CC', 'Rel_Never', and 'Relevance')
        rel_always = row["Rel_Always"]  # float
        rel_cc = row["Rel_CC"]  # float
        rel_never = row["Rel_Never"]  # float
        relevance = row["Relevance"]  # string
        # Combine them into a list
        other_measures_combination = [rel_always, rel_cc, rel_never, relevance]
        user_ratings.append(other_measures_combination)

    # Create a DataFrame with the combinations and also keep the original Barrier and Coping Option for group extraction
    combinations_df = pd.DataFrame({
        "Barrier": barriers,
        "Coping Option": coping_options,
        "Aggregated User Rating": user_ratings,
        "Original Barrier": df["barrier_group"],
        "Original Coping Group": df["Category"]
    })

    # Summarize results
    if verbose_mode:
        random_combinations = combinations_df.sample(10, random_state=42)
        print("10 random combinations:")
        print(random_combinations)
        unique_barriers = combinations_df["Barrier"].unique()
        unique_coping_options = combinations_df["Coping Option"].unique()
        logging.info(f"Number of unique barriers: {len(unique_barriers)}")
        logging.info(f"Number of unique coping options: {len(unique_coping_options)}")
        logging.info(f"Number of combinations: {len(combinations_df)} (max: {len(unique_barriers) * len(unique_coping_options)})")

    return combinations_df

def generate_LLM_ratings(combinations_df, max_workers=5, LLM="gpt-4o-mini", verbose_mode=False, test_mode=False):
    logging.info("Generating LLM scores for combinations using ThreadPoolExecutor...")

    responses = {}

    system_prompt = ("You are an expert in the field of interdisciplinary physical activity research. "
                     "You are tasked to evaluate the relevance of a certain 'coping option' for a specific 'barrier' ; The goal is to find appropriate coping options for barriers to ultimately increase daily physical activity levels. "
                     "You will provide three floats of which the total sum must ALWAYS be 1! ; The floats represent the evaluation you put on each evaluation label:"
                     "- 'Always relevant' (float) ; - 'Conditionally relevant' (float) ; - 'Never relevant' (float). --> so, these floats represent the evaluation of relevance itself. "
                     "Then, you will provide an integer that represents the overall relevance of the coping option for the barrier ; this is from 0 to 1000 (0=irrelevant, 1000=very relevant). "
                     "Finally, you will provide a binary response that summarizes whether the coping option can be considered relevant or not, for that specific barrier: 'relevant' or 'irrelevant'. "
                     "IMPORTANT: For the overall relevance (int), ensure to extremely detailled, therefore use increments of 1."
                     "The floats should also be precisely defined, so use increments of 0.001. ")

    if verbose_mode:
        print(f"System prompt: {system_prompt}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in combinations_df.iterrows():
            if test_mode and index >= 20:
                break

            barrier = row["Barrier"]
            coping_option = row["Coping Option"]
            user_query = f"Evaluate the relevance of the COPING OPTION '{coping_option}' for the BARRIER '{barrier}'."

            if verbose_mode:
                print(f"User query (Index {index}): {user_query}")

            def wrapped_call(index=index, user_query=user_query):
                result = call_GPT(
                    system_prompt=system_prompt,
                    user_query=user_query,
                    pydantic_model=Response_LLM,
                    model=LLM
                )
                logging.info(f"Processed item '{index + 1}/{len(combinations_df)}' for LLM-based relevance evaluation...")
                return result

            future = executor.submit(wrapped_call)
            responses[index] = future

    LLM_response_scores = [responses[i].result() for i in sorted(responses.keys())]

    if verbose_mode:
        for idx, response in enumerate(LLM_response_scores):
            print(f"LLM response for index {idx}: {response}")

    return LLM_response_scores

def compute_correlation_matrix(combinations_df, LLM_response_scores, full_output_path):
    """
    Build a correlation matrix based on user and LLM ratings across
    the correct barrier-coping option pairs.
    Data streams:
      - User ratings from 'Aggregated User Rating' (first three values)
      - LLM ratings from the response.
    """
    correlation_data = []
    num_pairs = len(LLM_response_scores)

    for i in range(num_pairs):
        user_ratings = combinations_df.loc[i, "Aggregated User Rating"]
        always_relevant_USER = float(user_ratings[0])
        conditionally_relevant_USER = float(user_ratings[1])
        never_relevant_USER = float(user_ratings[2])
        response = LLM_response_scores[i]
        always_relevant_LLM = response.nuanced_response.always_relevant
        conditionally_relevant_LLM = response.nuanced_response.conditionally_relevant
        never_relevant_LLM = response.nuanced_response.never_relevant
        overall_relevance_LLM = response.overall_relevance

        correlation_data.append({
            "always_USER": always_relevant_USER,
            "conditionally_USER": conditionally_relevant_USER,
            "never_USER": never_relevant_USER,
            "always_LLM": always_relevant_LLM,
            "conditionally_LLM": conditionally_relevant_LLM,
            "never_LLM": never_relevant_LLM,
            "overall_LLM": overall_relevance_LLM
        })

    # Create DataFrame from collected data and add group info
    corr_df = pd.DataFrame(correlation_data)
    # Add barrier and coping groups by extracting from the original combination strings
    corr_df['barrier_group'] = combinations_df["Barrier"].apply(extract_barrier_group)
    corr_df['coping_group'] = combinations_df["Coping Option"].apply(extract_coping_group)

    # Compute Pearson correlation matrix on numeric columns
    correlation_matrix = corr_df.drop(columns=["barrier_group", "coping_group"]).corr(method="pearson")

    print("\n--- Pearson Correlation Matrix ---")
    print(correlation_matrix)

    # Compute classification accuracy as:
    # (proportion of user ratings that are "relevant" minus proportion of LLM ratings that are "relevant") * 100
    total = num_pairs
    user_relevant_count = sum(1 for i in range(total) if str(combinations_df.loc[i, "Aggregated User Rating"][3]).strip().lower() == "relevant")
    llm_relevant_count = sum(1 for i in range(total) if str(LLM_response_scores[i].binary_response).strip().lower() == "relevant")
    classification_accuracy = ((user_relevant_count / total) - (llm_relevant_count / total)) * 100

    # Print the global classification accuracy to the console
    print(f"\n--- Global Classification Error: {classification_accuracy:.2f}% ---\n")

    # Fix title issue by using subplots to ensure the title is displayed
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix)
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45)
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)
    fig.colorbar(cax, ax=ax)
    ax.set_title(f"Pearson Correlation Matrix (Classification Error: {classification_accuracy:.2f}%)", pad=20)

    for (i, j), val in np.ndenumerate(correlation_matrix.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)

    fig.savefig(full_output_path, dpi=300)
    plt.show()

    return corr_df

def compute_grouped_correlation_matrices(corr_df, combinations_df, LLM_response_scores, base_image_dir):
    # Create subdirectories for separate groups
    barrier_groups_dir = os.path.join(base_image_dir, "separate", "barrier_groups")
    coping_groups_dir = os.path.join(base_image_dir, "separate", "coping_option_groups")
    os.makedirs(barrier_groups_dir, exist_ok=True)
    os.makedirs(coping_groups_dir, exist_ok=True)

    total = len(combinations_df)
    # Compute grouped correlation matrices for barrier groups
    barrier_groups = corr_df['barrier_group'].dropna().unique()
    for group in barrier_groups:
        group_df = corr_df[corr_df['barrier_group'] == group]
        if len(group_df) < 2:
            continue
        group_corr = group_df.drop(columns=["barrier_group", "coping_group"]).corr(method="pearson")
        # Compute classification accuracy for this barrier group
        indices = corr_df[corr_df['barrier_group'] == group].index
        user_rel_count = sum(1 for i in indices if str(combinations_df.loc[i, "Aggregated User Rating"][3]).strip().lower() == "relevant")
        llm_rel_count = sum(1 for i in indices if str(LLM_response_scores[i].binary_response).strip().lower() == "relevant")
        group_total = len(indices)
        if group_total > 0:
            group_accuracy = ((user_rel_count / group_total) - (llm_rel_count / group_total)) * 100
        else:
            group_accuracy = 0
        # Print domain classification accuracy to console
        print(f"Barrier Group '{group}' Classification Error: {group_accuracy:.2f}%")
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(group_corr)
        ax.set_xticks(range(len(group_corr.columns)))
        ax.set_xticklabels(group_corr.columns, rotation=45)
        ax.set_yticks(range(len(group_corr.index)))
        ax.set_yticklabels(group_corr.index)
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"Pearson Correlation Matrix for Barrier Group '{group}' (Classification Error: {group_accuracy:.2f}%)", pad=20)
        for (i, j), val in np.ndenumerate(group_corr.values):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)
        group_output_path = os.path.join(barrier_groups_dir, f"correlation_matrix_barrier_group_{group}.png")
        fig.savefig(group_output_path, dpi=300)
        plt.show()

    # Compute grouped correlation matrices for coping option groups
    coping_groups = corr_df['coping_group'].dropna().unique()
    for group in coping_groups:
        group_df = corr_df[corr_df['coping_group'] == group]
        if len(group_df) < 2:
            continue
        group_corr = group_df.drop(columns=["barrier_group", "coping_group"]).corr(method="pearson")
        indices = corr_df[corr_df['coping_group'] == group].index
        user_rel_count = sum(1 for i in indices if str(combinations_df.loc[i, "Aggregated User Rating"][3]).strip().lower() == "relevant")
        llm_rel_count = sum(1 for i in indices if str(LLM_response_scores[i].binary_response).strip().lower() == "relevant")
        group_total = len(indices)
        if group_total > 0:
            group_accuracy = ((user_rel_count / group_total) - (llm_rel_count / group_total)) * 100
        else:
            group_accuracy = 0
        # Print domain classification accuracy to console
        print(f"Coping Option Group '{group}' Classification Error: {group_accuracy:.2f}%")
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(group_corr)
        ax.set_xticks(range(len(group_corr.columns)))
        ax.set_xticklabels(group_corr.columns, rotation=45)
        ax.set_yticks(range(len(group_corr.index)))
        ax.set_yticklabels(group_corr.index)
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"Pearson Correlation Matrix for Coping Option Group '{group}' (Classification Error: {group_accuracy:.2f}%)", pad=20)
        for (i, j), val in np.ndenumerate(group_corr.values):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)
        group_output_path = os.path.join(coping_groups_dir, f"correlation_matrix_coping_group_{group}.png")
        fig.savefig(group_output_path, dpi=300)
        plt.show()

def save_llm_results_excel(combinations_df, LLM_response_scores, output_file="llm_results.xlsx"):
    """
    Save the raw LLM results into an Excel file in a pivoted format.
    Rows are barriers, columns are coping options, and each cell contains
    the raw LLM result including the binary response.
    """
    records = []
    num_pairs = len(LLM_response_scores)
    for i in range(num_pairs):
        barrier = combinations_df.loc[i, "Barrier"]
        coping_option = combinations_df.loc[i, "Coping Option"]
        response = LLM_response_scores[i]
        cell_str = (f"Always: {response.nuanced_response.always_relevant}, "
                    f"Cond.: {response.nuanced_response.conditionally_relevant}, "
                    f"Never: {response.nuanced_response.never_relevant}, "
                    f"Overall: {response.overall_relevance}, "
                    f"Binary: {response.binary_response}")
        records.append({"Barrier": barrier, "Coping Option": coping_option, "LLM Result": cell_str})
    raw_results_df = pd.DataFrame(records)
    pivot_df = raw_results_df.pivot(index="Barrier", columns="Coping Option", values="LLM Result")
    pivot_df.to_excel(output_file)
    logging.info(f"LLM raw results saved to {output_file}")

# ----------------------------------------------------------------------------#
# ---- Main function ---- #
# ----------------------------------------------------------------------------#

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.critical("No OpenAI API key found in environment variables.")
        raise ValueError("No OpenAI API key found in environment variables.")

    openai.api_key = api_key
    logger.info("OpenAI API key loaded successfully.")

    file_path = "//OSD_data/relevance/relevance_by_combination.csv"

    combinations_df = create_combinations(file_path, verbose_mode=True)

    LLM_response_scores = generate_LLM_ratings(combinations_df, max_workers=200, LLM="gpt-4o-mini", verbose_mode=False, test_mode=False)

    base_image_dir = "//results/images"
    full_dir = os.path.join(base_image_dir, "full")
    os.makedirs(full_dir, exist_ok=True)
    full_output_path = os.path.join(full_dir, "correlation_matrix_gpt4omini.png")

    corr_df = compute_correlation_matrix(combinations_df, LLM_response_scores, full_output_path)

    compute_grouped_correlation_matrices(corr_df, combinations_df, LLM_response_scores, base_image_dir)

    # Uncomment below lines to save LLM results to Excel if desired
    # results_base = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/Ondernemingen/PECAN2.0/Documenten/Comparison_UserRatings_LLMRatings/results"
    # os.makedirs(results_base, exist_ok=True)
    # output_file = os.path.join(results_base, "LLM_results.xlsx")
    # save_llm_results_excel(combinations_df, LLM_response_scores, output_file=output_file)

# note: classification accuracy refers to classification error
