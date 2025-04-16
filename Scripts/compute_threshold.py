import pandas as pd
import numpy as np

def compute_thresholds_and_deciles(file_path, n_boot=1000):
    """
    1) Print overall proportion of 'relevant' vs 'irrelevant' across the entire dataset.
    2) Empirically estimate the score where P(relevant)=0.5 for each of
       'Rel_Always', 'Rel_CC', 'Rel_Never' by interpolating between adjacent
       unique scores whose proportions straddle 0.5, with a 95% bootstrap CI.
    3) Bin scores into deciles and print, for each decile:
         - count of relevant
         - total count
         - empirical P(relevant)
    """
    df = pd.read_csv(file_path)
    df['relevant_bin'] = (df['Relevance'] == 'relevant').astype(int)

    # --- New: overall relevance proportions ---
    counts = df['Relevance'].value_counts()
    props = df['Relevance'].value_counts(normalize=True)
    print("Overall relevance proportions:")
    print(f"  relevant:   {counts.get('relevant', 0)} ({props.get('relevant', 0):.3f})")
    print(f"  irrelevant: {counts.get('irrelevant', 0)} ({props.get('irrelevant', 0):.3f})")

    def find_crossing_threshold(data, col):
        gp = data.groupby(col)['relevant_bin'].agg(['sum', 'count'])
        gp['prop'] = gp['sum'] / gp['count']
        gp = gp.sort_index()
        props = gp['prop'].values
        scores = gp.index.values
        for i in range(1, len(props)):
            if props[i-1] < 0.5 <= props[i]:
                x1, x2 = scores[i-1], scores[i]
                y1, y2 = props[i-1], props[i]
                return x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
        return np.nan

    thresholds = {}
    for col in ['Rel_Always', 'Rel_CC', 'Rel_Never']:
        sub = df[[col, 'relevant_bin']].dropna()
        boot_thr = []
        for _ in range(n_boot):
            samp = sub.sample(len(sub), replace=True)
            thr = find_crossing_threshold(samp, col)
            if np.isfinite(thr):
                boot_thr.append(thr)
        if boot_thr:
            est = np.mean(boot_thr)
            lo, hi = np.percentile(boot_thr, [2.5, 97.5])
        else:
            est = lo = hi = np.nan
        thresholds[col] = (est, lo, hi)

    # Print threshold estimates
    print("\nEmpirical thresholds (P(relevant)=0.5) with 95% CI:")
    for col, (est, lo, hi) in thresholds.items():
        print(f"  {col}: {est:.3f} (95% CI: {lo:.3f}–{hi:.3f})")

    # Print decile‐binned counts and proportions
    for col in ['Rel_Always', 'Rel_CC', 'Rel_Never']:
        print(f"\n--- Decile proportions for {col} ---")
        df[f'{col}_decile'] = pd.qcut(df[col], 10, labels=False) + 1
        prop = (
            df
            .groupby(f'{col}_decile')['relevant_bin']
            .agg(relevant='sum', total='count')
        )
        prop['P(relevant)'] = prop['relevant'] / prop['total']
        print(prop)

if __name__ == "__main__":
    file_path = "//OSD_data/relevance/relevance_by_combination.csv"
    compute_thresholds_and_deciles(file_path)
