import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset.csv")

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Salary"].fillna(df["Salary"].mean(), inplace=True)

country_encoder = LabelEncoder()
df["Country"] = country_encoder.fit_transform(df["Country"])

purchased_encoder = LabelEncoder()
df["Purchased"] = purchased_encoder.fit_transform(df["Purchased"])

X = df[["Country", "Age", "Salary"]]
y = df["Purchased"]

model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully")

new_sample = [[country_encoder.transform(["France"])[0], 40, 60000]]
prediction = model.predict(new_sample)

if prediction[0] == 1:
    print("Prediction: Yes")
else:
    print("Prediction: No")