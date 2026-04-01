# Autentykacja banknotów – drzewo decyzyjne

Projekt klasyfikacji banknotów jako **prawdziwe** (klasa 0) lub **sfałszowane** (klasa 1) na podstawie cech obrazów uzyskanych z transformaty falkowej. Wykorzystano klasyfikator **drzewa decyzyjnego** z biblioteki scikit-learn.

## Zbiór danych

Dane pochodzą z repozytorium [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication) (ID: 267). Zbiór zawiera **1372 próbki** opisane czterema cechami numerycznymi wyekstrahowanymi z obrazów banknotów:

| Cecha | Opis |
|---|---|
| `variance` | Wariancja transformaty falkowej |
| `skewness` | Skośność transformaty falkowej |
| `curtosis` | Kurtoza transformaty falkowej |
| `entropy` | Entropia obrazu |

### Rozkład klas

| Klasa | Znaczenie | Liczba próbek |
|---|---|---|
| 0 | Banknot prawdziwy | 762 |
| 1 | Banknot sfałszowany | 610 |

Zbiór jest stosunkowo zbalansowany (55.5% / 44.5%).

### Statystyki opisowe

| | variance | skewness | curtosis | entropy |
|---|---|---|---|---|
| Średnia | 0.434 | 1.922 | 1.398 | −1.192 |
| Odch. std. | 2.843 | 5.869 | 4.310 | 2.101 |
| Min | −7.042 | −13.773 | −5.286 | −8.548 |
| Max | 6.825 | 12.952 | 17.927 | 2.450 |

## Eksploracyjna analiza danych (EDA)

### Variance

Cecha o najsilniejszej zdolności dyskryminacyjnej - prawdziwe banknoty przyjmują wyraźnie wyższe wartości wariancji niż sfałszowane.

<p align="center">
  <img src="hist_variance.png" width="480" alt="Histogram cechy variance">
  <img src="hist_variance_by_class.png" width="480" alt="Histogram variance wg klasy">
</p>

### Skewness

Rozkład skośności jest wielomodalny. Klasy częściowo się nakładają, ale banknoty prawdziwe skupiają się w okolicach wartości dodatnich (5–10), a sfałszowane - ujemnych i bliskich zeru.

<p align="center">
  <img src="hist_skewness.png" width="480" alt="Histogram cechy skewness">
  <img src="hist_skewness_by_class.png" width="480" alt="Histogram skewness wg klasy">
</p>

### Curtosis (kurtoza)

Rozkład jest prawoskośny z długim ogonem. Obie klasy nakładają się mocno w przedziale −5…5, ale klasa 1 (fałszywki) częściej przyjmuje wartości skrajne (>10).

<p align="center">
  <img src="hist_curtosis.png" width="480" alt="Histogram cechy curtosis">
  <img src="hist_curtosis_by_class.png" width="480" alt="Histogram curtosis wg klasy">
</p>

### Entropy

Entropia jest skoncentrowana wokół 0 dla obu klas. Różnice między klasami są tu niewielkie - to cecha o najsłabszej sile dyskryminacyjnej.

<p align="center">
  <img src="hist_entropy.png" width="480" alt="Histogram cechy entropy">
  <img src="hist_entropy_by_class.png" width="480" alt="Histogram entropy wg klasy">
</p>

## Eksperymenty

### Walidacja krzyżowa (10-fold)

Zastosowano **stratyfikowaną 10-krotną walidację krzyżową** do zbadania wpływu hiperparametru `max_depth` na jakość klasyfikacji.

| max_depth | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| 2 | 0.908 | 0.918 | 0.872 | 0.894 |
| 3 | 0.936 | 0.935 | 0.920 | 0.927 |
| 4 | 0.952 | 0.942 | 0.951 | 0.946 |
| 5 | 0.970 | 0.975 | 0.957 | 0.966 |
| **6** | **0.984** | **0.982** | **0.982** | **0.982** |
| None | 0.983 | 0.977 | 0.984 | 0.980 |

Najwyższy F1-score uzyskano dla `max_depth = 6`. Dalsze zwiększanie głębokości (None = bez ograniczeń) nieznacznie obniża wynik, co sugeruje lekkie przeuczenie.

<p align="center">
  <img src="f1_vs_max_depth.png" width="560" alt="Wykres F1 vs max_depth">
</p>

### Ewaluacja na zbiorze testowym

Model z `max_depth = 6` wytrenowano na 80% danych i przetestowano na pozostałych 20% (podział stratyfikowany).

#### Macierz pomyłek

<p align="center">
  <img src="confusion_matrix_tree.png" width="480" alt="Macierz pomyłek">
</p>

Na 275 próbek testowych model popełnił jedynie **2 błędy** - oba to fałszywe pozytywy (prawdziwe banknoty zaklasyfikowane jako fałszywki). Daje to dokładność na poziomie **99.3%**.

### Struktura drzewa decyzyjnego

<p align="center">
  <img src="decision_tree_structure.png" width="100%" alt="Struktura drzewa decyzyjnego">
</p>

Korzeń drzewa dzieli próbki po cesze `variance ≤ 0.703` - co potwierdza, że wariancja jest najważniejszą cechą dyskryminacyjną w tym zbiorze.

## Struktura projektu

```
.
├── README.md                    # Ten plik
├── banknote_authentication.csv  # Zbiór danych
├── download_banknote.py         # Skrypt pobierania danych z UCI
├── banknote_eda.py              # Eksploracyjna analiza danych
├── banknote_experiments.py      # Walidacja krzyżowa i trening modelu
├── banknote_summary_stats.csv   # Statystyki opisowe
├── banknote_class_counts.csv    # Rozkład klas
├── cv_results_depth.csv         # Wyniki walidacji krzyżowej
├── hist_*.png                   # Histogramy cech
├── f1_vs_max_depth.png          # Wykres F1 vs max_depth
├── confusion_matrix_tree.png    # Macierz pomyłek
└── decision_tree_structure.png  # Wizualizacja drzewa
```

## Uruchomienie

```bash
pip install pandas scikit-learn matplotlib ucimlrepo

# 1. Pobranie danych (opcjonalne - CSV jest w repozytorium)
python download_banknote.py

# 2. Analiza eksploracyjna
python banknote_eda.py

# 3. Eksperymenty z drzewem decyzyjnym
python banknote_experiments.py
```

## Wymagania

- Python 3.10+
- pandas
- scikit-learn
- matplotlib
- ucimlrepo (tylko do pobrania danych)
