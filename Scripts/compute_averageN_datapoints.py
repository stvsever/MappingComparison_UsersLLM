import pandas as pd
import numpy as np

file_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/Ondernemingen/PECAN2.0/Documenten/Comparison_UserRatings_LLMRatings/OSF_data/relevance/relevance_by_combination.csv"

# Read the CSV file
df = pd.read_csv(file_path)


def compute_average_n_datapoints(df, column="Count_Total"):
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    values = df[column].dropna().astype(int)
    # Get descriptive statistics
    description = df["Count_Total"].dropna().astype(int).describe()
    print(description)

    return values


# Run the function and get the values
datapoints = compute_average_n_datapoints(df)
