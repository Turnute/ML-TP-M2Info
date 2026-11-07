from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


filepath = "TP2/.ressources/unzip/Wang"
image_size = (256, 256)
num_classes = 10


def load_images(filepath):
    # Tri alphabétique, comme les lignes de WangSignatures.xlsx selon l'énoncé.
    # Pour retrouver le même test que ex2.1, il faut les mêmes images dans
    # le même ordre : 0.jpg, 1.jpg, 10.jpg, 100.jpg, etc.
    filenames = sorted(
        path for path in Path(filepath).iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not filenames:
        raise ValueError(f"Aucune image trouvée dans {filepath}")

    # Allocation directe : évite une liste de 1000 grands tableaux à recopier.
    X = np.empty((len(filenames), *image_size, 3), dtype=np.float32)
    label = np.empty(len(filenames), dtype=np.int32)

    for i, path in enumerate(filenames):
        label[i] = int(path.stem) // 100
        if not 0 <= label[i] < num_classes:
            raise ValueError(f"Classe hors de 0..9 pour {path.name}")

        image = keras.utils.load_img(
            path, target_size=image_size, color_mode="rgb"
        )
        X[i] = keras.utils.img_to_array(image)
        image.close()

    # Les CNN entraînés de zéro reçoivent des pixels entre 0 et 1.
    X /= 255.0
    return X, label


def plot_history(history, name):
    # Même fonction que dans ex2.1.
    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Entraînement : {name}")

    ax1.plot(epochs, history.history["accuracy"], label="Training Accuracy")
    ax1.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history.history["loss"], label="Training Loss")
    ax2.plot(epochs, history.history["val_loss"], label="Validation Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()


def create_simple_cnn(input_shape):
    # Convolution puis classification
    return keras.models.Sequential([                    # RÔLE                                      # SORTIE
        layers.Input(shape=input_shape),                # Reçoit l'image                            # 256*256*3
        layers.Conv2D(32, (3, 3), activation="relu"),   # Applique 32 filtres (apprentissage)       # 254*254*32
        layers.MaxPooling2D(pool_size=(2, 2)),          # Garde le maximum de chaque zone 2x2       # 127*127*32
        layers.Flatten(),                               # Range toutes les valeurs dans un vecteur  # 516128
        layers.Dense(num_classes, activation="softmax") # Calcule les scores
    ])


def create_deep_cnn(input_shape, augmentation=False):
    model = keras.models.Sequential([
        layers.Input(shape=input_shape)
    ])

    if augmentation:
        # Ces transformations ne s'appliquent que pendant l'apprentissage.
        # 0.1 pour RandomRotation = +/- 36 degrés, pas +/- 0.1 degré.
        model.add(layers.RandomFlip("horizontal", seed=42))
        model.add(layers.RandomRotation(0.1, seed=43))
        model.add(layers.RandomZoom(0.1, seed=44))

    # Même idée que la boucle sur weights de ex2.1, avec des convolutions.
    for filters in (32, 64, 128):
        model.add(layers.Conv2D(
            filters, (3, 3), activation="relu", padding="same"
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Une moyenne par carte de caractéristiques : ici un vecteur de taille 128.
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def create_augmented_cnn(input_shape):
    # Même CNN profond, auquel on ajoute les transformations des images.
    return create_deep_cnn(input_shape, augmentation=True)


def create_transfer_model(input_shape):
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input

    # Le premier lancement peut télécharger les poids appris sur ImageNet.
    base_model = VGG16(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False

    # X est entre 0 et 1 pour TOUS nos appels à train_model.
    # VGG16 attend un prétraitement particulier de pixels entre 0 et 255.
    # On le fait dans le modèle pour ne pas changer les données à chaque appel.
    return keras.models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(255.0),
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])


def train_model(X, label, name, epoch=20, create_model=create_simple_cnn):
    # Même séparation 80/20 que ex2.1, avec la même graine et stratification.
    X_train, X_test, label_train, label_test = train_test_split(
        X,
        label,
        test_size=0.2,
        random_state=42,
        stratify=label
    )

    # La création du modèle a lieu APRÈS le réglage de la graine aléatoire.
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    model = create_model(X_train.shape[1:])  # (256, 256, 3), au lieu de (d,)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.summary()

    # Même encodage des labels que ex2.1.
    history = model.fit(
        X_train,
        to_categorical(label_train, num_classes=num_classes),
        epochs=epoch,
        batch_size=32,
        validation_split=0.2
    )

    score = model.evaluate(
        X_test,
        to_categorical(label_test, num_classes=num_classes),
        verbose=0
    )

    accuracy = score[1]
    print(f"{name}")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Taux d'erreur : {(1 - accuracy) * 100:.2f}%")
    print(f"Loss : {score[0]:.4f}")

    probabilites = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilites, axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        label_test,
        predictions,
        labels=np.arange(num_classes),
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"Matrice de confusion : {name}")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    fig.tight_layout()

    plot_history(history, name)


if __name__ == "__main__":
    X, label = load_images(filepath)
    print("Dimensions des images :", X.shape)
    print("Dimensions des labels :", label.shape)
    print("Effectifs par classe :", np.bincount(label, minlength=num_classes))

    # TEST 1 : CNN simple (actif par défaut).
    # train_model(X, label, "CNN simple", epoch=20)

    # TEST 2 : trois blocs convolution/pooling, moyenne globale et dropout.
    # train_model(X, label, "CNN profond", epoch=20,
    #             create_model=create_deep_cnn)

    # # TEST 3 : même CNN profond avec augmentation des données.
    train_model(X, label, "CNN profond + augmentation", epoch=50,
                create_model=create_augmented_cnn)

    # TEST 4 : extracteur VGG16 préentraîné et figé, nouveau classifieur.
    # train_model(X, label, "Transfert VGG16", epoch=20,
    #             create_model=create_transfer_model)

    plt.show()
