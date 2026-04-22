import pandas as pd
import pickle
from xgboost import XGBClassifier

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"

df = pd.read_csv(DATA_FILE)

X = df.drop("home_run", axis=1)
y = df["home_run"]

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
)

model.fit(X, y)

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
print("Rows:", len(df))
print("Average HR label rate:", round(y.mean(), 4))
print("Feature count:", X.shape[1])
print("Features:", list(X.columns))