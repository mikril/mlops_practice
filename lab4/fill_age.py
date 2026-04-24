import pandas as pd

df = pd.read_csv("titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df.to_csv("titanic.csv", index=False)
print(f"age_nan={df['Age'].isna().sum()}, age_mean={df['Age'].mean():.2f}")
