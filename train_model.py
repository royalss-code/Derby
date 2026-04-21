import pandas as pd
import pickle
from xgboost import XGBClassifier

df = pd.read_csv("data.csv")

X = df.drop("home_run", axis=1)
y = df["home_run"]

# 🔥 ONLY THIS PART CHANGED
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
print("Rows:", len(df))
print("Average HR label rate:", round(y.mean(), 4))
print("Features:", list(X.columns))