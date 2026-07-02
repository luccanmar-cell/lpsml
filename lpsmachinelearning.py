import pandas as pd

df = pd.read_excel("tarifa.xlsx",engine="openpyxl")

print(df.head())

print(df.isnull().sum())

split_df = df["Accesorios"].str.split(",", expand = True)
split_df.columns = [f"Value{i+1}" for i in range(split_df.shape[1])]
split_df = split_df.apply(pd.to_numeric, errors = "coerce").fillna(0)
df = pd.concat([df.drop(columns=["Accesorios"]), split_df], axis = 1)
print(df)

print(df.isnull().sum())

df.fillna(0, inplace=True)

print(df.isnull().sum())

df.to_excel("newtarifa.xlsx", index=False)
