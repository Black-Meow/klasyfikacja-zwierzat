
import json
import math
import random

from pathlib import Path

import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps, UnidentifiedImageError

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


# =====================================================
# 1. KONFIGURACJA PROJEKTU
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "animal_classifier.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_animal_classifier.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

SEED = 123

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Utworzenie katalogu na modele

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Ustawienie powtarzalności eksperymentu

tf.keras.utils.set_random_seed(SEED)

rng = random.Random(SEED)


# =====================================================
# 2. SPRAWDZENIE KATALOGU DATASET
# =====================================================

print("\n======================================")
print("     KLASYFIKACJA GATUNKÓW ZWIERZĄT")
print("======================================")

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Nie znaleziono katalogu: {DATASET_DIR}"
    )

class_dirs = sorted([
    folder
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
])

if len(class_dirs) < 2:
    raise ValueError(
        "Dataset musi zawierać przynajmniej "
        "dwa foldery z różnymi gatunkami."
    )

class_names = [
    folder.name
    for folder in class_dirs
]

print("\nRozpoznane gatunki:")

for name in class_names:
    print(f" - {name}")


# =====================================================
# 3. SPRAWDZANIE POPRAWNOŚCI ZDJĘĆ
# =====================================================

def is_valid_image(image_path):

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except (UnidentifiedImageError, OSError, ValueError):
        return False

    except Exception:
        return False


# =====================================================
# 4. AUTOMATYCZNY PODZIAŁ ZDJĘĆ
# =====================================================

def calculate_split_counts(total):

    if total < 3:
        raise ValueError(
            "Każdy gatunek musi posiadać "
            "przynajmniej 3 poprawne zdjęcia."
        )

    ratios = [
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO
    ]

    # Obliczenie dokładnych proporcji

    exact_counts = [
        total * ratio
        for ratio in ratios
    ]

    # Początkowe zaokrąglenie w dół

    counts = [
        math.floor(value)
        for value in exact_counts
    ]

    # Rozdysponowanie pozostałych zdjęć
    # zgodnie z największą częścią ułamkową

    remaining = total - sum(counts)

    remainders = [
        exact_counts[i] - counts[i]
        for i in range(3)
    ]

    order = sorted(
        range(3),
        key=lambda i: remainders[i],
        reverse=True
    )

    for i in order[:remaining]:
        counts[i] += 1

    # Zapewnienie przynajmniej jednego zdjęcia
    # w każdym zbiorze, jeśli liczba zdjęć na to pozwala

    for i in [1, 2]:

        if counts[i] == 0:

            donor = max(
                range(3),
                key=lambda j: counts[j] if j != i else -1
            )

            if counts[donor] <= 1:
                raise ValueError(
                    "Za mało zdjęć do utworzenia "
                    "trzech niepustych zbiorów."
                )

            counts[donor] -= 1
            counts[i] += 1

    return counts


# Listy przechowujące ścieżki do zdjęć
# oraz odpowiadające im numery klas

train_paths = []
train_labels = []

val_paths = []
val_labels = []

test_paths = []
test_labels = []


print("\n======================================")
print("        AUTOMATYCZNY PODZIAŁ")
print("======================================")


for label, folder in enumerate(class_dirs):

    # Pobranie zdjęć z katalogu danego gatunku

    all_images = sorted([
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_FORMATS
    ])

    # Sprawdzenie poprawności każdego zdjęcia

    valid_images = []
    invalid_count = 0

    for image_path in all_images:

        if is_valid_image(image_path):
            valid_images.append(str(image_path))
        else:
            invalid_count += 1

    total = len(valid_images)

    if total < 3:
        raise ValueError(
            f"Folder '{folder.name}' zawiera tylko "
            f"{total} poprawnych zdjęć. "
            "Potrzebne są minimum 3."
        )

    # Losowe wymieszanie zdjęć

    rng.shuffle(valid_images)

    # Obliczenie liczby zdjęć dla każdego zbioru

    train_count, val_count, test_count = (
        calculate_split_counts(total)
    )

    # Podział listy zdjęć

    train_images = valid_images[:train_count]

    val_images = valid_images[
        train_count:train_count + val_count
    ]

    test_images = valid_images[
        train_count + val_count:
    ]

    # Przypisanie zdjęć i etykiet do zbiorów

    train_paths.extend(train_images)
    train_labels.extend([label] * len(train_images))

    val_paths.extend(val_images)
    val_labels.extend([label] * len(val_images))

    test_paths.extend(test_images)
    test_labels.extend([label] * len(test_images))

    # Informacje o podziale

    print(f"\nGatunek: {folder.name}")
    print(f"  Wszystkie zdjęcia: {total}")
    print(f"  Train: {len(train_images)}")
    print(f"  Validation: {len(val_images)}")
    print(f"  Test: {len(test_images)}")

    if invalid_count > 0:
        print(f"  Pominięte uszkodzone pliki: {invalid_count}")

    if total < 20:
        print(
            "  UWAGA: Mała liczba zdjęć. "
            "Wyniki modelu mogą być niestabilne."
        )


# =====================================================
# 5. PODSUMOWANIE PODZIAŁU
# =====================================================

print("\n======================================")
print("         PODSUMOWANIE PODZIAŁU")
print("======================================")

total_images = (
    len(train_paths)
    + len(val_paths)
    + len(test_paths)
)

print(
    f"Train: {len(train_paths)} "
    f"({len(train_paths) / total_images * 100:.2f}%)"
)

print(
    f"Validation: {len(val_paths)} "
    f"({len(val_paths) / total_images * 100:.2f}%)"
)

print(
    f"Test: {len(test_paths)} "
    f"({len(test_paths) / total_images * 100:.2f}%)"
)

print(f"Łącznie zdjęć: {total_images}")


# =====================================================
# 6. WCZYTYWANIE I PRZYGOTOWANIE ZDJĘĆ
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image_with_pil(path_tensor):

    # Odczyt ścieżki przekazanej przez TensorFlow

    image_path = path_tensor.numpy().decode("utf-8")

    # Otwieranie zdjęcia

    with Image.open(image_path) as image:

        # Uwzględnienie orientacji EXIF
        # (ważne dla zdjęć z telefonów)

        image = ImageOps.exif_transpose(image)

        # Konwersja do RGB

        image = image.convert("RGB")

        # Zmiana rozmiaru zdjęcia

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


def load_image(path, label):

    # Wykorzystanie PIL do wczytania obrazu

    image = tf.py_function(
        func=load_image_with_pil,
        inp=[path],
        Tout=tf.float32
    )

    # Ustalenie wymiarów obrazu

    image.set_shape([
        IMG_SIZE,
        IMG_SIZE,
        3
    ])

    return image, label


# =====================================================
# 7. TWORZENIE ZBIORÓW TENSORFLOW
# =====================================================

def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    # Losowe mieszanie zdjęć treningowych

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    # Wczytywanie i przetwarzanie zdjęć

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    # Grupowanie zdjęć w partie

    dataset = dataset.batch(BATCH_SIZE)

    # Optymalizacja

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


print("\nPrzygotowywanie zbiorów TensorFlow...")

train_ds = create_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_dataset(
    val_paths,
    val_labels
)

test_ds = create_dataset(
    test_paths,
    test_labels
)

print("Zbiory danych zostały przygotowane!")


# =====================================================
# 8. AUGMENTACJA DANYCH
# =====================================================

print("\nKonfiguracja augmentacji danych...")

augmentation = keras.Sequential([

    # Odbicie lustrzane w poziomie

    layers.RandomFlip("horizontal"),

    # Losowy obrót od -180 do +180 stopni

    layers.RandomRotation(
        factor=0.5,
        fill_mode="reflect"
    ),

    # Losowe powiększanie i pomniejszanie

    layers.RandomZoom(
        height_factor=0.1,
        width_factor=0.1
    ),

], name="data_augmentation")


# =====================================================
# 9. TWORZENIE MODELU EFFICIENTNETB0
# =====================================================

print("\nŁadowanie EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Zamrożenie warstw modelu bazowego

base_model.trainable = False


# =====================================================
# 10. BUDOWANIE SIECI NEURONOWEJ
# =====================================================

inputs = keras.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Augmentacja zdjęć

x = augmentation(inputs)

# Analiza obrazu przez EfficientNetB0

# training=False utrzymuje warstwy
# Batch Normalization w trybie inference.

x = base_model(
    x,
    training=False
)

# Redukcja map cech do wektora

x = layers.GlobalAveragePooling2D()(x)

# Dropout ograniczający przeuczenie

x = layers.Dropout(0.3)(x)

# Warstwa klasyfikująca gatunki

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)


# =====================================================
# 11. KOMPILACJA MODELU
# =====================================================

print("\nKompilowanie modelu...")

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()


# =====================================================
# 12. CALLBACKI
# =====================================================

callbacks = [

    # Zapisywanie najlepszej wersji modelu

    keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    # Automatyczne zatrzymanie treningu,
    # gdy wyniki walidacyjne przestaną się poprawiać

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Zmniejszanie szybkości uczenia,
    # gdy model przestaje się poprawiać

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.000001,
        verbose=1
    )

]


# =====================================================
# 13. TRENOWANIE MODELU
# =====================================================

print("\n======================================")
print("       ROZPOCZYNAM TRENOWANIE")
print("======================================")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks

)


# =====================================================
# 14. TESTOWANIE MODELU
# =====================================================

print("\n======================================")
print("         TESTOWANIE MODELU")
print("======================================")

test_loss, test_accuracy = model.evaluate(
    test_ds
)

print("\nWYNIKI TESTOWANIA")

print(f"Test Loss: {test_loss:.4f}")

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# =====================================================
# 15. ZAPIS MODELU
# =====================================================

print("\nZapisywanie modelu...")

# EarlyStopping z restore_best_weights=True
# przywraca najlepsze wagi z treningu.

model.save(str(MODEL_PATH))


# =====================================================
# 16. ZAPIS NAZW KLAS
# =====================================================

print("Zapisywanie nazw gatunków...")

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        ensure_ascii=False,
        indent=4
    )


# =====================================================
# 17. ZAKOŃCZENIE
# =====================================================

print("\n======================================")
print("       TRENOWANIE ZAKOŃCZONE")
print("======================================")

print(f"Model: {MODEL_PATH}")
print(f"Najlepszy model: {BEST_MODEL_PATH}")
print(f"Lista klas: {CLASS_NAMES_PATH}")

print("\nMożesz teraz uruchomić app.py.")
