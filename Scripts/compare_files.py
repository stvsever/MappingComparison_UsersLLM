import pandas as pd


def compare_files(csv_file_path, excel_file_path, verbose_mode=False):
    # -----------------------------
    # Read and process CSV file
    # -----------------------------
    df_csv = pd.read_csv(csv_file_path)

    # Extract instance fields (convert to lowercase)
    csv_barriers = df_csv["Instance_barrier"].dropna().astype(str).str.lower().tolist()
    csv_coping = df_csv["Instance_solution"].dropna().astype(str).str.lower().tolist()

    # -----------------------------
    # Read and process Excel file
    # -----------------------------
    df_ex = pd.read_excel(excel_file_path)

    # For the Excel file:
    # - Barriers are in the first column (ignoring the header row)
    # - Coping options are in the header row from the second column onward.
    excel_barriers = df_ex.iloc[:, 0].dropna().astype(str).str.lower().tolist()
    excel_coping = [str(option).lower() for option in df_ex.columns[1:] if pd.notna(option)]

    # -----------------------------
    # Compare instance values
    # -----------------------------
    barriers_match = set(csv_barriers) == set(excel_barriers)
    coping_match = set(csv_coping) == set(excel_coping)

    if verbose_mode:
        print("CSV Barriers:", csv_barriers)
        print("Excel Barriers:", excel_barriers)
        print("CSV Coping Options:", csv_coping)
        print("Excel Coping Options:", excel_coping)

    print("\nComparison results:")
    print(f"Barriers match: {barriers_match}")
    if not barriers_match:
        diff_barriers = set(csv_barriers).symmetric_difference(set(excel_barriers))
        print("Differences in barriers:", diff_barriers)

    print(f"Coping options match: {coping_match}")
    if not coping_match:
        diff_coping = set(csv_coping).symmetric_difference(set(excel_coping))
        print("Differences in coping options:", diff_coping)

if __name__ == "__main__":
    # Define file paths
    csv_file_path = "//OSF_data/relevance/relevance_by_combination.csv"
    excel_file_path = "//Question_Matrix/MatrixInputR2.xlsx"

    # Compare the files and print the results in verbose mode
    compare_files(csv_file_path, excel_file_path, verbose_mode=True)
