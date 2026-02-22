import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("dataset.csv")

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Salary"].fillna(df["Salary"].mean(), inplace=True)

country_encoder = LabelEncoder()
df["Country"] = country_encoder.fit_transform(df["Country"])

purchased_encoder = LabelEncoder()
df["Purchased"] = purchased_encoder.fit_transform(df["Purchased"])

X = df[["Country", "Age", "Salary"]]
y = df["Salary"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

accuracy = r2_score(y, predictions)

print("Model trained successfully")
print("Accuracy (R2 Score):", accuracy)