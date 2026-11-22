# Koncepcja projektu — klasyfikacja dzikich ptaków

## KONCEPCJA PROJEKTU

**Klasyfikacja gatunków dzikich zwierząt na podstawie zdjęć**  
**Zakres projektu: rozpoznawanie gatunków dzikich ptaków**

### Członkowie zespołu

**Zespół nr 1**

- Ambroziak Mateusz
- Duda Paweł
- Klejmont Piotr
- Ramotowski Adam

**Akademia Łomżyńska**  
Wydziałowy Projekt Zespołowy / Informatyka

*Dokument koncepcyjny — wersja robocza*

---

## 1. Temat i krótki opis projektu

Tematem projektu jest opracowanie aplikacji internetowej wykorzystującej metody uczenia maszynowego do klasyfikowania gatunków dzikich ptaków na podstawie przesłanych zdjęć. Użytkownik będzie mógł wybrać plik graficzny, a system przeanalizuje obraz i wskaże najbardziej prawdopodobny gatunek spośród klas uwzględnionych w przygotowanym zbiorze danych.

Projekt ma charakter edukacyjny i demonstracyjny. Jego celem jest zapoznanie się z procesem przygotowania danych obrazowych, trenowania lub dostrajania modelu klasyfikacyjnego oraz integracji modelu z prostą aplikacją webową. Wynik działania systemu będzie przewidywaniem modelu i nie powinien być traktowany jako gwarantowane oznaczenie gatunku.

## 2. Główny cel i planowany rezultat

Celem głównym jest stworzenie prototypu aplikacji internetowej, która przyjmuje zdjęcie dzikiego ptaka i zwraca przewidywaną klasę. W miarę możliwości interfejs pokaże również wartość ufności predykcji.

Planowane rezultaty:

- przygotowany i opisany zbiór zdjęć przypisanych do określonych gatunków;
- model klasyfikacji obrazu oparty na konwolucyjnej sieci neuronowej (CNN) lub transfer learning;
- prostą aplikację internetową do przesyłania zdjęć i wyświetlania wyniku;
- podstawowe testy poprawności działania oraz ocenę jakości modelu.

## 3. Wybrany język programowania i technologie

Podstawowym językiem programowania będzie Python. Został wybrany ze względu na szerokie zastosowanie w uczeniu maszynowym, dostępność bibliotek do analizy obrazów oraz możliwość szybkiego przygotowania części serwerowej aplikacji.

| Obszar | Wstępna propozycja | Zastosowanie |
|---|---|---|
| Język programowania | Python | Implementacja przetwarzania danych, treningu modelu i backendu. |
| Uczenie maszynowe | PyTorch albo TensorFlow/Keras | Budowa i trenowanie CNN lub dostrajanie modelu wstępnie wytrenowanego. |
| Przetwarzanie obrazów | Pillow; opcjonalnie OpenCV | Odczyt, konwersja, zmiana rozmiaru i kontrola jakości zdjęć. |
| Analiza danych | NumPy, pandas | Operacje na etykietach, manifestach danych i wynikach eksperymentów. |
| Przygotowanie zbioru | scikit-learn | Podział danych i obliczanie metryk oceny klasyfikacji. |
| Aplikacja internetowa | Flask albo FastAPI | Obsługa przesyłania zdjęć i zwracania wyników klasyfikacji. |
| Interfejs użytkownika | HTML, CSS, opcjonalnie JavaScript | Formularz wyboru zdjęcia i prezentacja wyniku. |
| Testy | pytest | Automatyczne testy funkcji przetwarzania i endpointów. |
| Kontrola wersji | Git | Współpraca zespołu i historia zmian. |

## 4. Wstępna propozycja przetwarzania danych

Na początku zostanie wybrany zbiór zdjęć dzikich ptaków, który posiada etykiety gatunków i pozwala na wykorzystanie w projekcie zgodnie z warunkami licencji. Źródło danych nie zostało jeszcze ostatecznie ustalone. Przed rozpoczęciem pracy należy sprawdzić licencję, jakość opisów i liczbę zdjęć dla każdej klasy.

Planowane kroki:

1. pozyskanie zdjęć i zapisanie informacji o źródle oraz etykiecie każdego obrazu;
2. sprawdzenie, czy pliki można poprawnie odczytać, oraz usunięcie plików uszkodzonych;
3. ujednolicenie nazw klas i przygotowanie mapowania gatunków na etykiety numeryczne;
4. sprawdzenie duplikatów oraz nierównowagi liczby zdjęć między gatunkami;
5. zmiana rozmiaru zdjęć i normalizacja zgodnie z wymaganiami wybranego modelu;
6. podział danych na zbiór treningowy, walidacyjny i testowy;
7. opcjonalne zastosowanie augmentacji wyłącznie dla zbioru treningowego, np. niewielkich obrotów, odbić poziomych i zmian jasności.

Jako wstępny wariant można rozważyć podział 70% danych do treningu, 15% do walidacji i 15% do testów. Proporcje te zostaną potwierdzone po sprawdzeniu liczebności klas. Należy zapobiegać przeciekowi danych, czyli sytuacji, w której niemal identyczne zdjęcia trafiają jednocześnie do treningu i testu.

## 5. Wstępna propozycja modelu

Rozważone zostaną dwa warianty. Pierwszy to prosta konwolucyjna sieć neuronowa (CNN), trenowana na przygotowanych zdjęciach. Drugi to transfer learning, czyli wykorzystanie modelu wstępnie wytrenowanego na dużym zbiorze obrazów i dostosowanie jego warstwy klasyfikacyjnej do gatunków objętych projektem.

Transfer learning jest rekomendowany jako pierwszy wariant eksperymentalny, ponieważ może ograniczyć czas treningu i zapotrzebowanie na dużą liczbę oznaczonych zdjęć. Ostateczny wybór będzie zależał od wyników.

Jakość modelu zostanie oceniona na zbiorze testowym z wykorzystaniem m.in. accuracy, precision, recall, F1-score oraz macierzy pomyłek. Konkretne wartości zostaną podane dopiero po przeprowadzeniu eksperymentów.

## 6. Ogólny podział zadań w zespole

Podział ma charakter wstępny i może zostać zmieniony po ustaleniu składu zespołu oraz dostępnych kompetencji. W przypadku zespołu większego niż cztery osoby wybrane obszary można podzielić na mniejsze zadania.

| Rola / obszar | Zakres odpowiedzialności |
|---|---|
| Członek zespołu 1 — koordynacja i dokumentacja | Ustalenie zakresu, pilnowanie harmonogramu, zebranie wymagań i przygotowanie dokumentacji końcowej. |
| Członek zespołu 2 — dane | Wyszukanie źródła zdjęć, weryfikacja licencji, czyszczenie danych, przygotowanie etykiet i podziałów. |
| Członek zespołu 3 — model ML | Przygotowanie eksperymentów CNN/transfer learning, trenowanie modelu, ocena metryk i analiza błędów. |
| Członek zespołu 4 — aplikacja webowa | Przygotowanie interfejsu, backendu, obsługi plików i integracji z modelem. |
| Wspólne zadania | Przegląd kodu, integracja modułów, testy, demonstracja projektu i przygotowanie prezentacji. |

Po ustaleniu zespołu należy wpisać imiona i nazwiska oraz uzgodnić odpowiedzialność za poszczególne zadania. Każdy członek powinien mieć jasno określony rezultat pracy, a integracja i testy powinny być wykonywane wspólnie.

## 7. Wstępny plan realizacji

| Etap | Zakres | Rezultat |
|---|---|---|
| 1. Analiza i plan | Doprecyzowanie klas, wymagań, źródeł danych i technologii. | Uzgodniona specyfikacja. |
| 2. Przygotowanie danych | Pobranie, weryfikacja, etykietowanie i podział zdjęć. | Gotowe manifesty danych. |
| 3. Model | Baseline i eksperymenty z CNN/transfer learning. | Wytrenowany i oceniony model. |
| 4. Aplikacja | Backend, interfejs i integracja inferencji. | Prototyp aplikacji internetowej. |
| 5. Testy i dokumentacja | Testy funkcjonalne, metryki, instrukcja i opis ograniczeń. | Gotowy projekt do demonstracji. |

## 8. Decyzje do podjęcia na początku projektu

- ostateczny zbiór zdjęć i warunki licencyjne;
- lista gatunków objętych klasyfikacją;
- wybór jednego frameworka ML: PyTorch albo TensorFlow/Keras;
- wybór backendu: Flask albo FastAPI;
- wymagania sprzętowe i sposób trenowania modelu;
- terminy, nazwiska członków zespołu i szczegółowy harmonogram.

## Podsumowanie

Projekt zakłada stworzenie prototypu aplikacji internetowej do rozpoznawania dzikich ptaków na podstawie zdjęć. Python będzie podstawowym językiem programowania, a przetwarzanie obrazów i klasyfikacja zostaną zrealizowane z użyciem wybranego frameworka uczenia maszynowego. Wstępny plan obejmuje przygotowanie danych, porównanie modelu CNN z podejściem transfer learning, integrację z interfejsem webowym oraz ocenę jakości rozwiązania. Szczegółowe parametry, źródło danych i ostateczne narzędzia zostaną ustalone na początku realizacji.

---

*Dokument koncepcyjny — wersja robocza*
