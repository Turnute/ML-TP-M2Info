import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

filepath = "TP2/.ressources/WangSignatures.xlsx"  # Lien vers le fichier Excel

# On met les données dans des DataFrames
df_PHOG = pd.read_excel(filepath, sheet_name="WangSignaturesPHOG", header=None, index_col=0)
df_JCD = pd.read_excel(filepath, sheet_name="WangSignaturesJCD", header=None, index_col=0)
df_CEDD = pd.read_excel(filepath, sheet_name="WangSignaturesCEDD", header=None, index_col=0)
df_FCTH = pd.read_excel(filepath, sheet_name="WangSignaturesFCTH", header=None, index_col=0)
df_Fuzzy = pd.read_excel(filepath, sheet_name="WangSignaturesFuzzyColorHistogr", header=None, index_col=0)

# Vecteurs de données
X_PHOG = df_PHOG.values.astype(float)  # mesures (1000 x d)
X_JCD = df_JCD.values.astype(float)
X_CEDD = df_CEDD.values.astype(float)
X_FCTH = df_FCTH.values.astype(float)
X_Fuzzy = df_Fuzzy.values.astype(float)

# Vecteur de labels (identiques pour chaque jeu de données)
label = np.array([int(nom[:-4]) // 100 for nom in df_PHOG.index])


def plot_history(history, name):
    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Entraînement : {name}")

    # Évolution de l'accuracy
    ax1.plot(epochs, history.history["accuracy"], label="Training Accuracy")
    ax1.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True)

    # Évolution de la loss
    ax2.plot(epochs, history.history["loss"], label="Training Loss")
    ax2.plot(epochs, history.history["val_loss"], label="Validation Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()


def train_model(X, label, name, epoch=20, weights=(16, 8), act="relu"):
    # On fait une séparation 80/20 des données entre apprentissage et test
    X_train, X_test, label_train, label_test = train_test_split(
        X,
        label,
        test_size=0.2,
        random_state=42,
        stratify=label
    )

    from tensorflow import keras
    keras.utils.set_random_seed(42)

    num_classes = 10                # nombre de classes à détecter (ici 10 selon l'énoncé)
    input_space = X_train.shape[1]  # taille des données

    # Construction dynamique des couches cachées
    model = keras.models.Sequential([
        keras.layers.Input(shape=(input_space,))
    ])
    for w in weights:
        model.add(keras.layers.Dense(w, activation=act))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    from tensorflow.keras.utils import to_categorical

    history = model.fit(
        X_train,
        to_categorical(label_train),
        epochs=epoch,
        batch_size=32,
        validation_split=0.2
    )

    from sklearn.metrics import ConfusionMatrixDisplay

    score = model.evaluate(
        X_test,
        to_categorical(label_test),
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
        labels=np.arange(10),
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"Matrice de confusion : {name}")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    fig.tight_layout()

    plot_history(history, name)


# # TESTS : Comparaison des mesures
# jeux = {
#     "PHOG": X_PHOG,
#     "JCD": X_JCD,
#     "CEDD": X_CEDD,
#     "FCTH": X_FCTH,
#     "Fuzzy": X_Fuzzy,
#     "Concaténation des cinq": np.concatenate(
#         (X_PHOG, X_JCD, X_CEDD, X_FCTH, X_Fuzzy), axis=1
#     ),
#     "JCD + CEDD + FCTH": np.concatenate(
#         (X_JCD, X_CEDD, X_FCTH), axis=1
#     )
# }
#
# for nom, mesures in jeux.items():
#     train_model(mesures, label, nom)

# # TESTS : Comparaison des époques
# epoques = [20, 50, 100, 150]
# X = np.concatenate((X_JCD, X_CEDD, X_FCTH), axis=1)
# for e in epoques:
#     train_model(X, label, f"Nombre d'époques : {e}", e)

# # TESTS : Comparaison des jeux de neurones (2 couches cachées)
# weights = [[8, 4], [16, 8], [32, 16], [64, 32], [128, 64]]
# X = np.concatenate((X_JCD, X_CEDD, X_FCTH), axis=1)
# for w in weights:
#     train_model(X, label, f"Neurones : {w}", 20, w)

# # TESTS : Comparaison du nombre de couches cachées
# architectures = [[16], [16, 8], [16, 8, 8]]
# X = np.concatenate((X_JCD, X_CEDD, X_FCTH), axis=1)
# for arch in architectures:
#     train_model(X, label, f"Architecture : {arch}", 20, arch)

# # TESTS : Comparaison des fonctions d'activation
# fonctions = ["relu", "tanh", "sigmoid"]
# X = np.concatenate((X_JCD, X_CEDD, X_FCTH), axis=1)
# for f in fonctions:
#     train_model(X, label, f"Fonction d'activation : {f}", 20, [16], f)

# RESEAU FINAL
X = np.concatenate((X_JCD, X_CEDD, X_FCTH), axis=1)
train_model(X, label, "Réseau final", 20, [32], "tanh");

plt.show()
