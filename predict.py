
import json

from pathlib import Path

import numpy as np

from PIL import Image, ImageOps

from tensorflow import keras


# =====================================================
# 1. KONFIGURACJA
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "animal_classifier.keras"

CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

IMG_SIZE = 224

# Minimalny wynik softmax wymagany do uznania,
# że model rozpoznał jedno ze znanych zwierząt.

THRESHOLD = 0.50

# Czy stosować dodatkową analizę zdjęcia
# obróconego o 180 stopni?

USE_TTA = True


# =====================================================
# 2. SPRAWDZENIE PLIKÓW
# =====================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Nie znaleziono modelu: {MODEL_PATH}. "
        "Najpierw uruchom train.py."
    )


if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(
        f"Nie znaleziono pliku klas: {CLASS_NAMES_PATH}. "
        "Najpierw uruchom train.py."
    )


# =====================================================
# 3. WCZYTANIE MODELU
# =====================================================

print("Wczytywanie modelu...")

MODEL = keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

print("Model został wczytany!")


# =====================================================
# 4. WCZYTANIE NAZW GATUNKÓW
# =====================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    CLASSES = json.load(file)


if not isinstance(CLASSES, list) or len(CLASSES) < 2:

    raise ValueError(
        "Plik class_names.json zawiera nieprawidłową listę klas."
    )


# Sprawdzenie zgodności liczby klas z modelem

model_classes = MODEL.output_shape[-1]

if model_classes != len(CLASSES):

    raise ValueError(
        "Liczba klas w modelu nie zgadza się "
        "z liczbą nazw w class_names.json."
    )


print("\nModel rozpoznaje następujące gatunki:")

for animal in CLASSES:
    print("-", animal)


# =====================================================
# 5. PRZYGOTOWANIE ZDJĘCIA
# =====================================================

def prepare_image(path):

    image_path = Path(path)

    if not image_path.is_file():

        raise FileNotFoundError(
            f"Nie znaleziono zdjęcia: {image_path}"
        )

    # Otwieranie zdjęcia

    with Image.open(image_path) as image:

        # Korekta orientacji na podstawie EXIF

        image = ImageOps.exif_transpose(image)

        # Konwersja do RGB

        image = image.convert("RGB")

        # Zmiana rozmiaru

        image = image.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.BILINEAR
        )

        # Zamiana zdjęcia na tablicę liczb

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    return image_array


# =====================================================
# 6. FUNKCJA ROZPOZNAWANIA
# =====================================================

def predict_image(path, use_tta=USE_TTA):

    # Wczytanie i przygotowanie zdjęcia

    original = prepare_image(path)

    # -------------------------------------------------
    # TEST-TIME AUGMENTATION
    # -------------------------------------------------

    if use_tta:

        # Zdjęcie obrócone o 180 stopni

        rotated = np.rot90(
            original,
            k=2
        )

        # Zapewnienie ciągłego układu pamięci tablicy

        rotated = np.ascontiguousarray(rotated)

        # Połączenie dwóch wersji zdjęcia

        images = np.stack([
            original,
            rotated
        ])

    else:

        # Analiza tylko oryginalnego zdjęcia

        images = np.expand_dims(
            original,
            axis=0
        )

    # -------------------------------------------------
    # ANALIZA PRZEZ SIEĆ NEURONOWĄ
    # -------------------------------------------------

    predictions = MODEL.predict(
        images,
        verbose=0
    )

    # Uśrednienie wyników obu analiz

    probabilities = predictions.mean(axis=0)

    # -------------------------------------------------
    # WYBÓR NAJLEPSZYCH WYNIKÓW
    # -------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:min(3, len(CLASSES))]

    top3 = []

    for index in top_indices:

        top3.append({

            "class": CLASSES[int(index)],

            "probability": round(
                float(probabilities[index]) * 100,
                2
            )

        })

    # -------------------------------------------------
    # NAJBARDZIEJ PRAWDOPODOBNY GATUNEK
    # -------------------------------------------------

    best_index = int(
        np.argmax(probabilities)
    )

    best_animal = CLASSES[best_index]

    confidence = float(
        probabilities[best_index]
    )

    # -------------------------------------------------
    # SPRAWDZENIE PROGU PEWNOŚCI
    # -------------------------------------------------

    recognized = confidence >= THRESHOLD

    if recognized:

        message = "Rozpoznano zwierzę."

    else:

        message = (
            "Model nie osiągnął wymaganego "
            "progu pewności rozpoznania."
        )

    # -------------------------------------------------
    # ZWROT WYNIKÓW
    # -------------------------------------------------

    return {

        "recognized": recognized,

        "message": message,

        "animal": best_animal,

        "confidence": round(
            confidence * 100,
            2
        ),

        "top3": top3

    }


# =====================================================
# 7. SAMODZIELNE TESTOWANIE PROGRAMU
# =====================================================

if __name__ == "__main__":

    print("\n======================================")
    print("       ROZPOZNAWANIE ZWIERZĄT")
    print("======================================")

    image_path = input(
        "\nPodaj ścieżkę do zdjęcia: "
    ).strip().strip('"')

    try:

        result = predict_image(image_path)

        print("\n======================================")
        print("             WYNIKI")
        print("======================================")

        print("\nTOP 3 ROZPOZNAWANE GATUNKI:\n")

        for item in result["top3"]:

            print(
                f'{item["class"]}: '
                f'{item["probability"]:.2f}%'
            )

        print("\n--------------------------------------")

        if result["recognized"]:

            print(
                f'Rozpoznane zwierzę: {result["animal"]}'
            )

            print(
                f'Wynik modelu: {result["confidence"]:.2f}%'
            )

        else:

            print(result["message"])

            print(
                f'Najwyższy wynik modelu: '
                f'{result["confidence"]:.2f}%'
            )

        print("--------------------------------------")

    except (FileNotFoundError, OSError, ValueError) as error:

        print(f"\nBłąd: {error}")

    except Exception as error:

        print(f"\nNieoczekiwany błąd: {error}")
