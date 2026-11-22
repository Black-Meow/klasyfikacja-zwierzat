# Klasyfikacja zwierząt

Projekt umożliwiający klasyfikację zwierząt na podstawie przesłanego obrazu. Aplikacja wykorzystuje wytrenowany wcześniej model uczenia maszynowego oraz interfejs webowy oparty na Flask.

## Wymagania

Do uruchomienia projektu potrzebne są:

- Python 3
- TensorFlow
- Flask
- Pillow
- NumPy

## Instalacja bibliotek

Przed uruchomieniem aplikacji należy zainstalować wymagane biblioteki.

W terminalu, w głównym katalogu projektu, wykonaj:

```bash
pip install tensorflow flask pillow numpy
```

Możesz również zainstalować wszystkie zależności z pliku `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Uruchomienie aplikacji

Model klasyfikacyjny jest już wytrenowany i dołączony do projektu. Nie ma potrzeby ponownego przeprowadzania procesu trenowania.

Aby uruchomić program, wystarczy:

1. Zainstalować wymagane biblioteki.
2. Otworzyć terminal w głównym katalogu projektu.
3. Uruchomić aplikację poleceniem:

```bash
python app.py
```

Po uruchomieniu aplikacja będzie dostępna pod adresem lokalnym wyświetlonym w terminalu (domyślnie Flask używa `http://127.0.0.1:5000`).

## Model

Projekt korzysta z wcześniej wytrenowanego modelu, który służy do rozpoznawania i klasyfikowania zwierząt na podstawie obrazów.

**Ważne:** model nie wymaga ponownego trenowania — jest gotowy do wykorzystania po uruchomieniu aplikacji.

## Technologie

- Python
- TensorFlow
- Flask
- Pillow
- NumPy
