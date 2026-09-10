from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import load_data, prepare_xy

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "exasens.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Run `python src/download_data.py` first."
    )

df = load_data(DATA_PATH)

print("=" * 70)
print("DATA AUDIT")
print("=" * 70)
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())
print("\nTarget distribution:")
print(df["Diagnosis"].value_counts(dropna=False))

# EDA: class distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Diagnosis", order=df["Diagnosis"].value_counts().index)
plt.title("Respiratory Condition Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(RESULTS / "class_distribution.png", dpi=200)
plt.close()

# EDA: age distribution by class
if "Age" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Diagnosis", y="Age")
    plt.title("Age Distribution by Respiratory Condition")
    plt.tight_layout()
    plt.savefig(RESULTS / "age_by_diagnosis.png", dpi=200)
    plt.close()

# EDA: missingness
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if len(missing):
    plt.figure(figsize=(9, 5))
    missing.plot(kind="bar")
    plt.title("Missing Values by Feature")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "missing_values.png", dpi=200)
    plt.close()

# Outlier report using IQR on numeric columns.
numeric_cols = df.select_dtypes(include="number").columns
outlier_rows = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        count = 0
    else:
        count = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    outlier_rows.append({"feature": col, "iqr_outliers": int(count)})
pd.DataFrame(outlier_rows).to_csv(RESULTS / "outlier_report.csv", index=False)

# Prepare data with leakage-safe preprocessing.
X, y, preprocessor = prepare_xy(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42
    ),
    "SVM RBF": SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}

cv_rows = []

for name, model in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    row = {"model": name}
    for metric in scoring:
        row[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
        row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
    cv_rows.append(row)

cv_results = pd.DataFrame(cv_rows).sort_values(
    "cv_f1_macro_mean", ascending=False
)
cv_results.to_csv(RESULTS / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(cv_results.round(4).to_string(index=False))

best_name = cv_results.iloc[0]["model"]
best_model = models[best_name]

final_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", best_model),
    ]
)
final_pipe.fit(X_train, y_train)

pred = final_pipe.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
    "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
}
pd.DataFrame([metrics]).to_csv(RESULTS / "final_metrics.csv", index=False)

print("\nBEST MODEL:", best_name)
print("\nFINAL TEST METRICS")
for k, v in metrics.items():
    print(f"{k:20s}: {v:.4f}")

print("\nCLASSIFICATION REPORT")
report = classification_report(y_test, pred, zero_division=0)
print(report)
(RESULTS / "classification_report.txt").write_text(report, encoding="utf-8")

labels = sorted(y.unique())
cm = confusion_matrix(y_test, pred, labels=labels)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=200)
plt.close()

print(f"\nSaved all results to: {RESULTS}")
