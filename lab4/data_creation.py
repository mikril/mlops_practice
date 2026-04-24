from catboost.datasets import titanic

train, _ = titanic()
df = train[["Pclass", "Sex", "Age"]]
df.to_csv("titanic.csv", index=False)
print(df.head())
print(f"rows={len(df)}, age_nan={df['Age'].isna().sum()}")
