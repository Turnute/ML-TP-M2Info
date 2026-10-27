import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

filepath = "TP2/.ressources/WangSignatures.xlsx"

df_PHOG = pd.read_excel(filepath, sheet_name = "WangSignaturesPHOG", header=None, index_col=0)
df_JCD = pd.read_excel(filepath, sheet_name = "WangSignaturesJCD", header=None, index_col=0 )
df_CEDD = pd.read_excel(filepath, sheet_name = "WangSignaturesCEDD", header=None, index_col=0 )
df_FCTH = pd.read_excel(filepath, sheet_name = "WangSignaturesFCTH", header=None, index_col=0 )
df_Fuzzy = pd.read_excel(filepath, sheet_name = "WangSignaturesFuzzyColorHistogr", header=None, index_col=0 )

X_PHOG = df_PHOG.values.astype(float) # mesures (1000 x d)
X_JCD = df_JCD.values.astype(float)
X_CEDD = df_CEDD.values.astype(float)
X_FCTH = df_FCTH.values.astype(float)
X_Fuzzy = df_Fuzzy.values.astype(float)
label = np.array([int(nom[:-4]) // 100 for nom in df_PHOG.index])

def train_model(X, label):
    # On fait une séparation 80/20 des données entre apprentissage et test 
    X_train, X_test, label_train, label_test = train_test_split(
        X,
        label,
        test_size=0.2,
        random_state=42,
        stratify=label
    )

    from tensorflow import keras
    num_classes = 10
    input_space = X_train.shape[1]

    model = keras.models.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(input_space,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

    from tensorflow.keras.utils import to_categorical

    history = model.fit(
        X_train,
        to_categorical(label_train),
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    score = model.evaluate(
        X_test,
        to_categorical(label_test)
    )

print("TRAIN PHOG")
train_model(X_PHOG, label)
print("TRAIN JCD")
train_model(X_JCD, label)
print("TRAIN CEDD")
train_model(X_CEDD, label)
print("TRAIN FCTH")
train_model(X_FCTH, label)
print("TRAIN Fuzzy")
train_model(X_Fuzzy, label)

X_concat = np.concatenate((X_PHOG, X_JCD, X_CEDD, X_FCTH, X_Fuzzy), axis=1)
print("TRAIN Concatenated")
train_model(X_concat, label)