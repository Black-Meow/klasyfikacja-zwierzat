# Wyniki trenowania modelu

## 1. Informacje ogólne

Model klasyfikacji obrazów zwierząt został wytrenowany przez 10 epok. W trakcie treningu rejestrowano wartość błędu treningowego (Train Loss), Log Loss oraz Accuracy.

## 2. Wyniki poszczególnych epok

| Epoka | Train Loss | Log Loss | Accuracy |
|---:|---:|---:|---:|
| 1 | 0.9486 | 1.8008 | 43.45% |
| 2 | 0.5587 | 1.8356 | 46.00% |
| 3 | 0.4305 | 1.7557 | 45.26% |
| 4 | 0.3636 | 2.0927 | 41.61% |
| 5 | 0.3084 | 2.0211 | 47.39% |
| 6 | 0.2785 | 2.0086 | 46.31% |
| 7 | 0.1845 | 1.9437 | 47.05% |
| 8 | 0.1627 | 2.1066 | 45.66% |
| 9 | 0.1532 | 2.0665 | 46.71% |
| 10 | 0.1317 | 2.0867 | 46.45% |

## 3. Analiza wyników

W trakcie kolejnych epok wartość Train Loss systematycznie malała. Oznacza to, że model coraz lepiej dopasowywał się do danych treningowych.

Najwyższą dokładność uzyskano w 5. epoce i wyniosła ona 47,39%.

Najniższą wartość Log Loss, wynoszącą 1,7557, uzyskano w 3. epoce.

Pomimo poprawy wyników na zbiorze treningowym nie zaobserwowano równie wyraźnej poprawy pozostałych metryk. Może to wskazywać na występowanie zjawiska przeuczenia (overfitting).

## 4. Wnioski

Uzyskane wyniki pokazują, że model nauczył się rozpoznawać część zależności występujących w obrazach, jednak jego skuteczność wymaga dalszej poprawy.

W kolejnych etapach projektu warto rozważyć:
- dostosowanie parametrów treningowych,
- zastosowanie dodatkowej augmentacji obrazów,
- eksperymenty z liczbą epok,
- analizę błędnie sklasyfikowanych obrazów,
- porównanie wyników różnych architektur modeli.

**Uwaga:** Wyniki pochodzą z lokalnego procesu trenowania i nie są oficjalnym wynikiem konkursu DrivenData. Log Loss i Accuracy opisano zgodnie z nazwami zapisanymi w pliku historii treningu.
