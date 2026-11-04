# Conser-vision: Klasyfikacja obrazów dzikich zwierząt

Projekt z zakresu widzenia komputerowego, wykorzystujący PyTorch, realizowany na podstawie konkursu [DrivenData – Conser-vision Practice Area](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/).

Celem projektu jest klasyfikowanie zdjęć z fotopułapek do ośmiu rozłącznych klas:

`antelope_duiker`, `bird`, `blank`, `civet_genet`, `hog`, `leopard`, `monkey_prosimian`, `rodent`.

> **Uwaga dotycząca zakresu:** pierwotny dokument koncepcyjny projektu uczelnianego koncentrował się na dzikich ptakach. Konkurs obejmuje jednak siedem grup dzikich zwierząt oraz klasę `blank` (puste zdjęcie), dlatego niniejsze repozytorium jest zgodne z oficjalną definicją konkursu. Model jest prototypem edukacyjnym, a nie systemem gwarantującym poprawną identyfikację gatunków.

## Zastosowane podejście

- **Framework:** PyTorch i torchvision.
- **Model bazowy:** ResNet18 z wagami wstępnie wytrenowanymi na ImageNet (transfer learning).
- **Walidacja:** podział grupowany według lokalizacji fotopułapek (`site`).
- **Metryka:** wieloklasowy Log Loss.
- **Wynik:** prawdopodobieństwa przynależności obrazu do poszczególnych klas, zapisane w formacie wymaganym przez DrivenData.

Podział walidacyjny jest grupowany według lokalizacji (`site`), ponieważ zbiór testowy konkursu zawiera lokalizacje, których nie ma w zbiorze treningowym. Losowy podział pojedynczych zdjęć mógłby prowadzić do zbyt optymistycznej oceny modelu.

## Struktura repozytorium

```text
conservision-wildlife-classifier/
├── data/
│   └── raw/                 # dane konkursowe (lokalnie)
├── models/                  # zapisane modele i historia treningu
├── submissions/             # wygenerowane pliki CSV
├── docs/
│   └── wyniki.md            # wyniki trenowania modelu
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_submission.py
├── .gitignore
├── requirements.txt
└── README.md
```

Foldery `data/`, `models/` i `submissions/` są przeznaczone do lokalnego przechowywania danych i wyników. Nie należy przesyłać danych konkursowych ani dużych plików modelu do repozytorium.

## 1. Pobranie danych

1. Dołącz do konkursu i zaakceptuj zasady korzystania z danych na platformie DrivenData.
2. Pobierz archiwum konkursowe z zakładki **Data**.
3. Rozpakuj pliki do katalogu `data/raw/`.

Oczekiwana struktura:

```text
data/raw/
├── train_features.csv
├── train_labels.csv
├── test_features.csv
├── submission_format.csv
├── train_features/
│   ├── ZJ000000.jpg
│   └── ...
└── test_features/
    ├── ZJ016556.jpg
    └── ...
```

Plik `submission_format.csv` należy uwzględnić, jeśli znajduje się w pobranym archiwum konkursowym.

Dane konkursowe są wykluczone z repozytorium Git. Nie przesyłaj zdjęć ani prywatnych plików konkursowych do GitHuba.

## 2. Instalacja

Zalecana wersja środowiska: **Python 3.12**.

Utworzenie środowiska wirtualnego:

```bash
python -m venv .venv
```

Aktywacja środowiska w Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Aktywacja w macOS / Linux:

```bash
source .venv/bin/activate
```

Instalacja wymaganych bibliotek:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Trenowanie modelu

Polecenia należy wykonywać z głównego katalogu repozytorium.

Przykładowe uruchomienie treningu:

```bash
python -m src.train --data-dir data/raw --epochs 10 --batch-size 32
```

Szybki test poprawności uruchomienia:

```bash
python -m src.train --data-dir data/raw --epochs 1 --batch-size 8 --device cpu
```

Najlepszy checkpoint modelu oraz historia treningu są zapisywane w katalogu `models/`.

Przy pierwszym uruchomieniu pobierane są publicznie dostępne wagi ImageNet dla modelu ResNet18. Jeśli nie masz dostępu do internetu, można zmienić inicjalizację modelu w `src/train.py`, ustawiając `weights=None`.

## 4. Generowanie predykcji

Po zakończeniu treningu uruchom:

```bash
python -m src.predict --data-dir data/raw --checkpoint models/best_model.pt
```

Skrypt generuje plik `submissions/submission.csv`.

Plik zawiera prawdopodobieństwa dla poszczególnych klas. Skrypt zachowuje kolejność identyfikatorów ze zbioru testowego i sprawdza, czy suma prawdopodobieństw w każdym wierszu wynosi w przybliżeniu 1.

Gotowy plik CSV można przesłać na stronie DrivenData w zakładce **Submit**.

## 5. Testy

Testy projektu można uruchomić poleceniem:

```bash
pytest -q
```

## 6. Wyniki i analiza

Wyniki procesu trenowania, w tym wartości Train Loss, Log Loss i Accuracy dla kolejnych epok, zostały przedstawione w pliku:

[`docs/wyniki.md`](docs/wyniki.md).

## Aktualne ograniczenia i dalsze kroki

Projekt stanowi pierwszy model bazowy (baseline). Nie jest deklaracją osiągnięcia wysokiego wyniku w rankingu konkursowym.

Możliwe kierunki dalszego rozwoju:

1. Porównanie ResNet18 z modelami EfficientNet lub ConvNeXt.
2. Dostosowanie rozmiaru obrazu, współczynnika uczenia i augmentacji.
3. Zastosowanie walidacji krzyżowej z grupowaniem według `site`.
4. Analiza Log Loss dla poszczególnych klas oraz macierzy pomyłek.
5. Przetestowanie augmentacji podczas predykcji (test-time augmentation) i kalibracji prawdopodobieństw.

Przed wyborem ostatecznego modelu należy zapisywać wyniki walidacji. Nie należy używać etykiet ze zbioru testowego konkursu, ponieważ nie są one udostępniane.

## Dane i konkurs

Zbiór danych jest udostępniany za pośrednictwem DrivenData we współpracy z Wild Chimpanzee Foundation oraz Max Planck Institute for Evolutionary Anthropology.

Dane należy pobierać bezpośrednio ze strony konkursu i wykorzystywać zgodnie z określonymi tam warunkami.

**Konkurs:** [DrivenData – Conser-vision Practice Area](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/)
