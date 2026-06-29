import pandas as pd

# Load saved model comparison results
results_df = pd.read_csv("results.csv")

# Sort by best accuracy
results_df = results_df.sort_values(
    "Best_Accuracy",
    ascending=False
).reset_index(drop=True)

# Best model
best_row = results_df.loc[
    results_df["Best_Accuracy"].idxmax()
]
