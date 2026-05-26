import pandas as pd

print("--- OMNIGUARD: DATA DETECTIVE ---")

# We are only going to read the first 5 rows to see what is going on
df = pd.read_csv('combined_text_dataset.csv', nrows=5)

print("\n1. Do you have a header row? Here are the column names Pandas sees:")
print(df.columns.tolist())

print("\n2. Here is EXACTLY what is inside your first row of data:")
print(df.iloc[0].values)