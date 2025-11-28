import joblib
import pathlib

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame):
    df = df.dropna()
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def main():
    data_path = pathlib.Path("data/raw_data.csv")
    model_dir = pathlib.Path("artifacts")
    model_dir.mkdir(exist_ok=True)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    (model_dir / "metrics.json").write_text(str(metrics))


if __name__ == "__main__":
    main()

