import pandas as pd
import pickle
from xgboost import XGBClassifier

df = pd.read_csv("data.csv")

X = df.drop("home_run", axis=1)
y = df["home_run"]

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved")
print("Average HR label rate:", y.mean())
