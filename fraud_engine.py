
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def detect_fraud(df):
    """
    Detect potentially fraudulent transactions using
    Isolation Forest and optional supervised learning.
    """

    df = df.copy()

    required_columns = [
        "amount",
        "transaction_count",
        "account_age_days",
        "is_foreign",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    features = df[required_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            contamination=0.10,
            random_state=42
        )),
    ])

    predictions = model.fit_predict(features)

    df["fraud_score"] = -model.decision_function(features)
    df["fraud_flag"] = predictions == -1
    df["review_flag"] = df["fraud_flag"]

    # Optional supervised model when labeled data is available.
    if "is_fraud" in df.columns:
        labels = pd.to_numeric(
            df["is_fraud"], errors="coerce"
        )

        valid = labels.isin([0, 1])

        if valid.sum() >= 10 and labels[valid].nunique() == 2:
            supervised = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced"
                )),
            ])

            supervised.fit(
                features.loc[valid],
                labels.loc[valid].astype(int)
            )

            fraud_class_index = list(
                supervised.named_steps["classifier"].classes_
            ).index(1)

            probabilities = supervised.predict_proba(features)[
                :, fraud_class_index
            ]

            df["fraud_probability"] = probabilities
            df["review_flag"] = (
                df["fraud_flag"] | (probabilities >= 0.70)
            )

    return df
