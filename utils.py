"""Utilitaires partagés par toutes les pages du portfolio.

IMPORTANT : les fichiers .pkl ont été sauvegardés depuis un notebook, donc les
fonctions / classes personnalisées qu'ils contiennent sont référencées sous le nom
``__main__.<nom>``. Pour pouvoir les recharger dans Streamlit, on les redéfinit ici
avec EXACTEMENT les mêmes noms et le même code, puis on les rattache à ``__main__``
juste avant le chargement (voir ``_register_pickle_symbols``).
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"


# --------------------------------------------------------------------------- #
# Transformers personnalisés (copiés des notebooks, ne pas renommer !)
# --------------------------------------------------------------------------- #
def discretize_pdays(X_in):
    """Bank marketing : découpe `pdays` en deux modalités (seuil trouvé par un arbre)."""
    valeurs = X_in.iloc[:, 0] if isinstance(X_in, pd.DataFrame) else X_in.ravel()
    bins = [-np.inf, 513.0, np.inf]
    labels = ["recent", "far_or_never"]
    return pd.cut(valeurs, bins=bins, labels=labels).to_frame()


def clean_str(series):
    """Prix des téléphones : normalisation des adresses (minuscules, sans accents…)."""
    series = series.astype(str).str.lower()
    series = series.apply(
        lambda x: "".join(c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn")
    )
    series = series.str.replace(r"[^\w\s]", " ", regex=True)
    series = series.str.replace(r"\s+", " ", regex=True).str.strip()
    return series


def clean_str_array(X):
    if isinstance(X, pd.DataFrame):
        s = X.iloc[:, 0]
    elif hasattr(X, "ndim") and X.ndim == 2:
        s = X[:, 0]
    else:
        s = X

    cleaned_series = clean_str(s)
    return cleaned_series.to_numpy().reshape(-1, 1)


class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    """Regroupe les modalités rares (< threshold %) dans une modalité `fill_value`."""

    def __init__(self, threshold=4.0, fill_value="other"):
        self.threshold = threshold
        self.fill_value = fill_value
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.frequent_categories_ = {}

        for col in X_df.columns:
            freq = X_df[col].value_counts(normalize=True) * 100

            frequent = set(freq[freq >= self.threshold].index)
            self.frequent_categories_[col] = frequent

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()

        for col in X_df.columns:
            frequent = self.frequent_categories_.get(col, set())
            X_df[col] = X_df[col].apply(lambda x: x if x in frequent else self.fill_value)

        return X_df.values if isinstance(X, list) else X_df


def _register_pickle_symbols() -> None:
    """Rattache les symboles personnalisés à `__main__` pour que joblib les retrouve."""
    main = sys.modules["__main__"]
    for obj in (discretize_pdays, clean_str, clean_str_array, RareCategoryEncoder):
        setattr(main, obj.__name__, obj)


# --------------------------------------------------------------------------- #
# Chargement des modèles (mis en cache : chargés une seule fois)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_pipeline(filename: str):
    """Charge un pipeline scikit-learn (.pkl) depuis le dossier models/."""
    _register_pickle_symbols()
    return joblib.load(MODELS_DIR / filename)


@st.cache_resource(show_spinner="Chargement du réseau de neurones…")
def load_cnn(filename: str = "CnnModel.h5"):
    """Charge le CNN Keras. L'import de TensorFlow est volontairement tardif :
    les autres pages restent rapides et n'en ont pas besoin."""
    from tensorflow.keras.models import load_model

    return load_model(MODELS_DIR / filename, compile=False)


# --------------------------------------------------------------------------- #
# Petits helpers d'affichage
# --------------------------------------------------------------------------- #
def fcfa(value: float) -> str:
    """12345678 -> '12 345 678 FCFA'."""
    return f"{value:,.0f}".replace(",", " ") + " FCFA"


def sidebar_footer(github_url: str | None = None) -> None:
    with st.sidebar:
        st.markdown("---")
        st.caption("Portfolio Machine Learning · Streamlit")
        if github_url:
            st.markdown(f"[Code source sur GitHub]({github_url})")
