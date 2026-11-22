
import uuid
import logging
import webbrowser

from pathlib import Path
from threading import Timer

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from PIL import Image, ImageOps, UnidentifiedImageError


# =====================================================
# 1. KONFIGURACJA APLIKACJI
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

TEMPLATE_FOLDER = BASE_DIR / "templates"
STATIC_FOLDER = BASE_DIR / "static"

UPLOAD_FOLDER = STATIC_FOLDER / "uploads"

# Utworzenie katalogów, jeżeli nie istnieją

TEMPLATE_FOLDER.mkdir(parents=True, exist_ok=True)
STATIC_FOLDER.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# Utworzenie aplikacji Flask

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_FOLDER),
    static_folder=str(STATIC_FOLDER)
)


# =====================================================
# 2. USTAWIENIA PRZESYŁANIA PLIKÓW
# =====================================================

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Maksymalny rozmiar przesyłanego zdjęcia: 10 MB

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Maksymalna liczba pikseli zdjęcia

MAX_IMAGE_PIXELS = 25_000_000

# Dozwolone formaty i odpowiadające im rozszerzenia

ALLOWED_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP"
}


# =====================================================
# 3. WCZYTANIE MODELU
# =====================================================

# Model jest wczytywany przez predict.py.
# Nie wczytujemy go ponownie przy każdym przesłaniu zdjęcia.

try:

    from predict import predict_image

    PREDICTOR_AVAILABLE = True
    PREDICTOR_ERROR = None

    print("\nModel klasyfikacyjny został załadowany.")

except Exception as error:

    predict_image = None

    PREDICTOR_AVAILABLE = False
    PREDICTOR_ERROR = error

    print("\nUWAGA: Nie udało się wczytać modelu.")
    print("Najpierw uruchom train.py.")
    print(f"Szczegóły: {error}")


# =====================================================
# 4. NAZWY GATUNKÓW DO WYŚWIETLANIA
# =====================================================

# Nazwy folderów mogą być zapisane bez polskich znaków.
# Tutaj zamieniamy je na czytelne nazwy w interfejsie.

DISPLAY_NAMES = {

    "dzik": "Dzik",

    "jelen": "Jeleń",

    "lis": "Lis",

    "niedzwiedz": "Niedźwiedź",

    "sarna": "Sarna",

    "wilk": "Wilk"

}


def display_animal_name(name):

    if not name:
        return "Nieznany gatunek"

    key = str(name).lower()

    return DISPLAY_NAMES.get(
        key,
        str(name).replace("_", " ").capitalize()
    )


# =====================================================
# 5. PRZYGOTOWANIE WYNIKÓW DO WYŚWIETLENIA
# =====================================================

def prepare_result(result):

    result = dict(result)

    # Czytelna nazwa najlepszego wyniku

    result["animal_label"] = display_animal_name(
        result.get("animal")
    )

    # Przygotowanie nazw gatunków w TOP 3

    formatted_top3 = []

    for item in result.get("top3", []):

        item = dict(item)

        item["label"] = display_animal_name(
            item.get("class")
        )

        formatted_top3.append(item)

    result["top3"] = formatted_top3

    return result


# =====================================================
# 6. SPRAWDZANIE POPRAWNOŚCI ZDJĘCIA
# =====================================================

def validate_image(filepath, extension):

    try:

        # Pierwsze otwarcie służy do sprawdzenia
        # struktury pliku.

        with Image.open(filepath) as image:

            detected_format = image.format

            width, height = image.size

            image.verify()

        # Sprawdzenie rzeczywistego formatu

        expected_format = ALLOWED_FORMATS.get(extension)

        if detected_format != expected_format:

            raise ValueError(
                "Rozszerzenie pliku nie odpowiada "
                "rzeczywistemu formatowi zdjęcia."
            )

        # Kontrola rozdzielczości

        if width <= 0 or height <= 0:

            raise ValueError(
                "Zdjęcie posiada nieprawidłowe wymiary."
            )

        if width * height > MAX_IMAGE_PIXELS:

            raise ValueError(
                "Rozdzielczość zdjęcia jest zbyt duża."
            )

        # Drugie otwarcie sprawdza możliwość
        # pełnego odczytania danych obrazu.

        with Image.open(filepath) as image:

            image.load()

            # Sprawdzenie, czy obraz można przekonwertować
            # na format RGB.

            image = ImageOps.exif_transpose(image)
            image.convert("RGB")

        return True

    except UnidentifiedImageError:

        raise ValueError(
            "Przesłany plik nie jest prawidłowym zdjęciem."
        )

    except Image.DecompressionBombError:

        raise ValueError(
            "Zdjęcie posiada zbyt dużą rozdzielczość."
        )

    except OSError:

        raise ValueError(
            "Nie udało się odczytać przesłanego zdjęcia."
        )


# =====================================================
# 7. STRONA GŁÓWNA
# =====================================================

@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html"
    )


# =====================================================
# 8. ANALIZA PRZESŁANEGO ZDJĘCIA
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Sprawdzenie, czy model jest dostępny

    if not PREDICTOR_AVAILABLE:

        app.logger.error(
            "Model nie jest dostępny: %s",
            PREDICTOR_ERROR
        )

        return render_template(
            "index.html",
            error=(
                "Model nie jest dostępny. "
                "Najpierw uruchom train.py, "
                "aby wytrenować i zapisać model."
            )
        ), 503

    # Pobranie przesłanego pliku

    file = request.files.get("image")

    if file is None or file.filename == "":

        return render_template(
            "index.html",
            error="Nie wybrano żadnego zdjęcia."
        ), 400

    # Oczyszczenie nazwy pliku

    safe_name = secure_filename(file.filename)

    # Pobranie rozszerzenia

    extension = Path(safe_name).suffix.lower()

    # Sprawdzenie rozszerzenia

    if extension not in ALLOWED_FORMATS:

        return render_template(
            "index.html",
            error=(
                "Nieobsługiwany format zdjęcia. "
                "Wybierz JPG, JPEG, PNG, WEBP lub BMP."
            )
        ), 400

    # Generowanie losowej nazwy.
    # Zapobiega to nadpisywaniu poprzednich zdjęć.

    filename = f"{uuid.uuid4().hex}{extension}"

    filepath = UPLOAD_FOLDER / filename

    try:

        # =============================================
        # ZAPISANIE ZDJĘCIA
        # =============================================

        file.save(filepath)

        if filepath.stat().st_size == 0:

            raise ValueError(
                "Przesłany plik jest pusty."
            )

        # =============================================
        # SPRAWDZENIE POPRAWNOŚCI ZDJĘCIA
        # =============================================

        validate_image(
            filepath,
            extension
        )

        # =============================================
        # ROZPOZNAWANIE ZWIERZĘCIA
        # =============================================

        app.logger.info(
            "Rozpoczynam analizę zdjęcia: %s",
            filename
        )

        result = predict_image(
            str(filepath)
        )

        # Przygotowanie nazw do interfejsu

        result = prepare_result(result)

        app.logger.info(
            "Zakończono analizę zdjęcia: %s",
            filename
        )

        # =============================================
        # WYŚWIETLENIE WYNIKÓW
        # =============================================

        return render_template(

            "index.html",

            image=filename,

            result=result

        )

    except ValueError as error:

        # Usunięcie niepoprawnego przesłanego pliku

        filepath.unlink(missing_ok=True)

        return render_template(
            "index.html",
            error=str(error)
        ), 400

    except Exception:

        # Usunięcie pliku w przypadku błędu analizy

        filepath.unlink(missing_ok=True)

        app.logger.exception(
            "Wystąpił błąd podczas analizy zdjęcia."
        )

        return render_template(
            "index.html",
            error=(
                "Wystąpił nieoczekiwany błąd "
                "podczas analizowania zdjęcia. "
                "Sprawdź komunikaty w terminalu."
            )
        ), 500


# =====================================================
# 9. OBSŁUGA ZBYT DUŻYCH PLIKÓW
# =====================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error=(
            "Przesłane zdjęcie jest zbyt duże. "
            "Maksymalny rozmiar pliku wynosi 10 MB."
        )
    ), 413


# =====================================================
# 10. AUTOMATYCZNE OTWIERANIE PRZEGLĄDARKI
# =====================================================

def open_browser():

    webbrowser.open_new_tab(
        "http://127.0.0.1:5050"
    )


# =====================================================
# 11. URUCHOMIENIE APLIKACJI
# =====================================================

if __name__ == "__main__":

    print("\n======================================")
    print("       ANIMAL DETECTOR")
    print("======================================")

    print("\nUruchamianie lokalnego serwera...")

    print("Adres: http://127.0.0.1:5050")

    # Automatyczne otwarcie przeglądarki

    Timer(
        1.5,
        open_browser
    ).start()

    # Uruchomienie Flask

    app.run(

        host="127.0.0.1",

        port=5050,

        debug=False,

        use_reloader=False,

        threaded=True

    )
