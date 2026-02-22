import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset.csv")

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Salary"].fillna(df["Salary"].mean(), inplace=True)

country_encoder = LabelEncoder()
df["Country"] = country_encoder.fit_transform(df["Country"])

purchased_encoder = LabelEncoder()
df["Purchased"] = purchased_encoder.fit_transform(df["Purchased"])

X = df[["Country", "Age", "Salary"]]
y = df["Purchased"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

print("Decision Tree model trained successfully")
print("Accuracy:", accuracy)