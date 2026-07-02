import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import StandardScaler

df = pd.read_excel("C:/Users/LuccaMarinaro/Downloads/tarifa.xlsx",engine="openpyxl")
#df = pd.read_excel("C:/Users/LuccaMarinaro/Downloads/tarifa.xlsx", sheet_name="Sheet1")

# Modify your data framework
# df.loc[df["Status"] == "Pending", "Status"] = "Completed"


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

# Save the changes back to Excel
df.to_excel("C:/Users/LuccaMarinaro/Downloads/tarifa.xlsx", index=False)

#df = df.apply(pd.to_numeric)

#X = df.drop(columns=["Prima"])
#Y = df["Prima"]

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#model = RandomForestRegressor()
#model.fit(X_train, Y_train)

#print("Accuracy: ", model.score(X_test, Y_test))