import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Llegir dades
df = pd.read_csv("dades_entrenament.csv")

lletres_valides = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
numeros_valids = [str(i) for i in range(10)]

df_lletres = df[df["label"].isin(lletres_valides)]
df_numeros = df[df["label"].isin(numeros_valids)]

print(f"Mostres totals: {len(df)}")
print(f"Mostres de lletres: {len(df_lletres)}")
print(f"Mostres de números: {len(df_numeros)}")


def augmentar_dades(X, y, factor=5, soroll=0.02):
    """Genera còpies amb soroll gaussià per augmentar el dataset."""
    X_aug = [X]
    y_aug = [y]
    for _ in range(factor - 1):
        X_soroll = X + np.random.normal(0, soroll, X.shape)
        X_aug.append(X_soroll)
        y_aug.append(y)
    return np.vstack(X_aug), np.concatenate(y_aug)


# ---------- LLETRES ----------
if len(df_lletres) > 0:
    X = df_lletres.drop("label", axis=1).to_numpy(dtype=np.float64)
    y = df_lletres["label"].to_numpy()
    
    # Augmentar
    X_aug, y_aug = augmentar_dades(X, y, factor=5, soroll=0.02)
    print(f"Lletres després d'augmentar: {len(X_aug)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
    
    clf_lletres = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=3000,
        learning_rate_init=0.001,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42
    )
    clf_lletres.fit(X_train, y_train)
    
    pred = clf_lletres.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Precisió LLETRES: {acc*100:.2f}%")
    joblib.dump(clf_lletres, "model_lletres.pkl")
    print("Model de LLETRES desat a model_lletres.pkl")
else:
    print("Avís: no hi ha dades de lletres.")


# ---------- NÚMEROS ----------
if len(df_numeros) > 0:
    X = df_numeros.drop("label", axis=1).to_numpy(dtype=np.float64)
    y = df_numeros["label"].to_numpy()
    
    # Augmentar MÉS els números perquè en tenim menys
    X_aug, y_aug = augmentar_dades(X, y, factor=8, soroll=0.025)
    print(f"Números després d'augmentar: {len(X_aug)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
    
    clf_numeros = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=3000,
        learning_rate_init=0.001,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42
    )
    clf_numeros.fit(X_train, y_train)
    
    pred = clf_numeros.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Precisió NÚMEROS: {acc*100:.2f}%")
    
    # Mostrar on falla (matriu de confusió)
    labels_unics = sorted(set(y))
    cm = confusion_matrix(y_test, pred, labels=labels_unics)
    print("\nMatriu de confusió NÚMEROS:")
    print("     " + " ".join(f"{l:>4}" for l in labels_unics))
    for i, label in enumerate(labels_unics):
        print(f"{label:>4} " + " ".join(f"{cm[i,j]:>4}" for j in range(len(labels_unics))))
    
    joblib.dump(clf_numeros, "model_numeros.pkl")
    print("\nModel de NÚMEROS desat a model_numeros.pkl")
else:
    print("Avís: no hi ha dades de números.")

print("\nEntrenament completat!")