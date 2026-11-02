[README.md](https://github.com/user-attachments/files/34294578/README_PL.md)
# Conser-vision: Klasyfikacja obrazów dzikich zwierząt

Projekt z zakresu widzenia komputerowego, wykorzystujący PyTorch [DrivenData Conser-vision Practice Area](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/).

Celem jest klasyfikowanie zdjęć z fotopułapek do ośmiu rozłącznych klas:

`antelope_duiker`, `bird`, `blank`, `civet_genet`, `hog`, `leopard`, `monkey_prosimian`, `rodent`.

> **Uwaga dotycząca zakresu:** pierwotny dokument koncepcyjny projektu uczelnianego koncentrował się na dzikich ptakach. Konkurs obejmuje jednak siedem grup dzikich zwierząt oraz klasę `blank` (puste zdjęcie), dlatego niniejsze repozytorium jest zgodne z oficjalną definicją konkursu. Model jest prototypem edukacyjnym, a nie systemem gwarantującym poprawną identyfikację gatunków.

## Zastosowane podejście

- **Framework:** PyTorch + torchvision
- **Model bazowy:** ResNet18 z wagami wstępnie wytrenowanymi na ImageNet (transfer learning)
- **Walidacja:** wydzielenie grup lokalizacji fotopułapek (`site`) w celu ograniczenia przecieku informacji związanej z lokalizacją
- **Metryka:** wieloklasowy log loss (logarytmiczna funkcja straty)
- **Wynik:** prawdopodobieństwa klas w dokładnym formacie wymaganym przez DrivenData

Podział walidacyjny jest grupowany według lokalizacji (`site`), ponieważ zbiór testowy konkursu zawiera lokalizacje, których nie ma w zbiorze treningowym. Losowy podział pojedynczych zdjęć może prowadzić do zbyt optymistycznej oceny modelu.

## Struktura repozytorium

```text
conservision-wildlife-classifier/
├── data/
│   └── raw/                 # dane konkursowe (nie są dodawane do repozytorium)
├── models/                  # wytrenowane checkpointy (nie są dodawane do repozytorium)
├── submissions/             # wygenerowane pliki CSV (nie są dodawane do repozytorium)
├── src/
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_submission.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Pobranie danych

1. Dołącz do konkursu i zaakceptuj zasady korzystania z danych na platformie DrivenData.
2. Pobierz archiwum konkursowe z zakładki **Data**.
3. Rozpakuj pliki tak, aby struktura katalogów wyglądała następująco:

```text
data/raw/
├── train_features.csv
├── train_labels.csv
├── test_features.csv
├── submission_format.csv       # jeśli plik znajduje się w pobranym archiwum
├── train_features/
│   ├── ZJ000000.jpg
│   └── ...
└── test_features/
    ├── ZJ016556.jpg
    └── ...
```

Dane konkursowe są celowo wykluczone z tego repozytorium Git. Nie przesyłaj zdjęć ani prywatnych plików konkursowych do GitHuba.

## 2. Instalacja

Zalecany jest Python w wersji 3.10 lub nowszej.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Instalacja zależności:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Trenowanie modelu

Polecenia należy wykonywać z głównego katalogu repozytorium:

```bash
python -m src.train --data-dir data/raw --epochs 10 --batch-size 32
```

Aby szybko sprawdzić, czy proces uruchamia się poprawnie:

```bash
python -m src.train --data-dir data/raw --epochs 1 --batch-size 8 --device cpu
```

Najlepszy checkpoint modelu oraz metadane treningu są zapisywane w katalogu `models/`.

Przy pierwszym uruchomieniu pobierane są publicznie dostępne wagi ImageNet dla modelu ResNet18. Jeśli nie masz dostępu do internetu, zmień inicjalizację modelu w pliku `src/train.py`, ustawiając `weights=None`.

## 4. Generowanie predykcji

Po zakończeniu treningu uruchom:

```bash
python -m src.predict --data-dir data/raw --checkpoint models/best_model.pt
```

Skrypt utworzy plik `submissions/submission.csv`. Zapisuje on jedno prawdopodobieństwo dla każdej klasy, zachowuje kolejność identyfikatorów ze zbioru testowego i sprawdza, czy suma prawdopodobieństw w każdym wierszu wynosi w przybliżeniu 1.

Prześlij ten plik CSV na stronie konkursu DrivenData, w zakładce **Submit**.

## 5. Testy

```bash
pytest -q
```

## Aktualne ograniczenia i dalsze kroki

Jest to pierwszy, odtwarzalny model bazowy — nie jest to deklaracja osiągnięcia wysokiego wyniku w rankingu konkursowym. Przydatne kierunki dalszych eksperymentów:

1. Porównanie ResNet18 z modelami EfficientNet lub ConvNeXt.
2. Dostosowanie rozmiaru obrazu, współczynnika uczenia i augmentacji.
3. Zastosowanie walidacji krzyżowej z grupowaniem według `site`.
4. Analiza log loss dla poszczególnych klas oraz macierzy pomyłek.
5. Przetestowanie augmentacji podczas predykcji (test-time augmentation) i kalibracji prawdopodobieństw.

Przed wyborem ostatecznego modelu zapisuj wszystkie wyniki walidacji. Nigdy nie używaj etykiet ze zbioru testowego konkursu — nie są one udostępniane.

## Dane i konkurs

Zbiór danych jest udostępniany za pośrednictwem DrivenData we współpracy z Wild Chimpanzee Foundation oraz Max Planck Institute for Evolutionary Anthropology. Pobieraj dane bezpośrednio ze strony konkursu i przestrzegaj określonych tam warunków ich wykorzystania.

Konkurs: https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/
