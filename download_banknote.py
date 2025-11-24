from ucimlrepo import fetch_ucirepo
import pandas as pd

def download_and_save_banknote_csv(
    out_path: str = "banknote_authentication.csv",
) -> None:
    banknote = fetch_ucirepo(id=267)

    X = banknote.data.features      # cechy
    y = banknote.data.targets       # etykiety

    df = pd.concat([X, y], axis=1)

    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    download_and_save_banknote_csv()
