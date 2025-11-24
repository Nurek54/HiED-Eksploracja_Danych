from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "banknote_authentication.csv"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


def run_cross_validation() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    features = ["variance", "skewness", "curtosis", "entropy"]
    X = df[features]
    y = df["class"]

    # 10-krotna walidacja krzyżowa z zachowaniem proporcji klas
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # zakres badanych wartości hiperparametru max_depth
    depth_grid = [2, 3, 4, 5, 6, None]

    records: list[dict] = []

    for depth in depth_grid:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)

        scores = cross_validate(
            clf,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
            return_train_score=False,
        )

        records.append(
            {
                "max_depth": "None" if depth is None else depth,
                "accuracy_mean": np.mean(scores["test_accuracy"]),
                "precision_mean": np.mean(scores["test_precision"]),
                "recall_mean": np.mean(scores["test_recall"]),
                "f1_mean": np.mean(scores["test_f1"]),
            }
        )

    results_df = pd.DataFrame(records)
    results_path = BASE_DIR / "cv_results_depth.csv"
    results_df.to_csv(results_path, index=False)

    print("Wyniki CV (średnie z 10-fold):")
    print(results_df)
    print(f"\nZapisano do pliku: {results_path}")

    fig, ax = plt.subplots()
    ax.plot(results_df["max_depth"].astype(str), results_df["f1_mean"], marker="o")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("F1-score (średnia z 10-fold CV)")
    ax.set_title("Wpływ hiperparametru max_depth na F1")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "f1_vs_max_depth.png", dpi=150)
    plt.close(fig)

    return results_df


def train_best_and_evaluate(results_df: pd.DataFrame) -> None:
    df = pd.read_csv(DATA_PATH)

    features = ["variance", "skewness", "curtosis", "entropy"]
    X = df[features]
    y = df["class"]

    # 20% danych jako test, z zachowaniem proporcji klas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_row = results_df.sort_values("f1_mean", ascending=False).iloc[0]
    best_depth_label = best_row["max_depth"]
    best_depth = None if best_depth_label == "None" else int(best_depth_label)

    print(
        f"\nNajlepszy max_depth wg F1: "
        f"{best_depth_label} (F1={best_row['f1_mean']:.3f})"
    )

    clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\nMacierz pomyłek (test):")
    print(cm)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Macierz pomyłek – drzewo decyzyjne")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix_tree.png", dpi=150)
    plt.close(fig)

    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(
        clf,
        feature_names=features,
        class_names=["real (0)", "fake (1)"],
        filled=True,
        impurity=True,
        rounded=True,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "decision_tree_structure.png", dpi=200)
    plt.close(fig)

    print(f"\nWygenerowano graf drzewa i macierz pomyłek w katalogu: {PLOTS_DIR}")


def main() -> None:
    results_df = run_cross_validation()
    train_best_and_evaluate(results_df)


if __name__ == "__main__":
    main()
