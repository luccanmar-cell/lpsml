import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("C:/Users/LuccaMarinaro/Downloads/newtarifa.xlsx",engine="openpyxl")

df = df.apply(pd.to_numeric)

X = df.drop(columns=["Prima"])
Y = df["Prima"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor()
model.fit(X_train, Y_train)
