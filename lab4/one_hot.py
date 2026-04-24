import pandas as pd

df = pd.read_csv("titanic.csv")
df = pd.get_dummies(df, columns=["Sex"], dtype=int)
df.to_csv("titanic.csv", index=False)
print(df.head())
