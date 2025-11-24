from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "banknote_authentication.csv"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("Podgląd danych:")
    print(df.head(), "\n")

    print("Informacje o ramce danych:")
    print(df.info(), "\n")

    print("Rozkład klas:")
    print(df["class"].value_counts(), "\n")

    desc = df.describe()
    print("Statystyki opisowe cech:")
    print(desc, "\n")
    desc.to_csv(BASE_DIR / "banknote_summary_stats.csv")

    class_counts = df["class"].value_counts().rename("count")
    class_counts.to_csv(BASE_DIR / "banknote_class_counts.csv")

    features = ["variance", "skewness", "curtosis", "entropy"]

    for col in features:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=30)
        ax.set_title(f"Histogram cechy {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Liczba obserwacji")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"hist_{col}.png", dpi=150)
        plt.close(fig)

    for col in features:
        fig, ax = plt.subplots()
        for label in [0, 1]:
            subset = df[df["class"] == label][col]
            ax.hist(subset, bins=30, alpha=0.5, label=f"class={label}")
        ax.set_title(f"Histogram cechy {col} wg klasy")
        ax.set_xlabel(col)
        ax.set_ylabel("Liczba obserwacji")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"hist_{col}_by_class.png", dpi=150)
        plt.close(fig)

    print(f"Histogramy zapisane w katalogu: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
