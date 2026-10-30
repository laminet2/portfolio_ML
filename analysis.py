"""Fonctions d'analyse (sans Streamlit) pour la page Exploration & ACP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import DATA_DIR

LOG_COLS = ["capital.gain", "capital.loss", "fnlwgt"]


def load_adult() -> tuple[pd.DataFrame, int] | None:
    """Charge le jeu Adult Census depuis data/ et applique le nettoyage du notebook.

    - data/adult.csv        : fichier brut (Kaggle) -> doublons supprimés, `education.num`
                              supprimée (redondante avec `education`), log1p sur 3 variables.
    - data/cleaned_df.csv   : fichier déjà nettoyé (export du notebook) -> utilisé tel quel.
    Retourne (DataFrame, nb_doublons_supprimes) ou None si aucun fichier n'est présent.
    """
    raw, cleaned = DATA_DIR / "adult.csv", DATA_DIR / "cleaned_df.csv"
    if raw.exists():
        df = pd.read_csv(raw)
        df.columns = df.columns.str.strip().str.replace("-", ".", regex=False)
        obj = df.select_dtypes(include="object").columns
        df[obj] = df[obj].apply(lambda s: s.str.strip())
        n_dup = int(df.duplicated().sum())
        df = df.drop_duplicates().drop(columns=["education.num"], errors="ignore")
        for col in LOG_COLS:
            df[col] = np.log1p(df[col])
        return df.reset_index(drop=True), n_dup
    if cleaned.exists():
        df = pd.read_csv(cleaned)
        df.columns = df.columns.str.strip().str.replace("-", ".", regex=False)
        return df, 0
    return None


def association_tests(df: pd.DataFrame, target: str = "income") -> pd.DataFrame:
    """Mann-Whitney (numériques) et chi-deux (catégorielles) contre la variable cible."""
    rows = []
    classes = sorted(df[target].unique())
    for col in df.select_dtypes(include="number").columns:
        g1 = df.loc[df[target] == classes[0], col]
        g2 = df.loc[df[target] == classes[1], col]
        _, p = stats.mannwhitneyu(g1, g2)
        rows.append({"Variable": col, "Type": "numérique", "Test": "Mann-Whitney U", "Statistique": np.nan, "p-value": p})
    for col in df.select_dtypes(include="object").columns.drop(target):
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[col], df[target]))
        rows.append({"Variable": col, "Type": "catégorielle", "Test": "Chi-deux", "Statistique": chi2, "p-value": p})
    out = pd.DataFrame(rows)
    out["Lien significatif (5 %)"] = np.where(out["p-value"] < 0.05, "oui", "non")
    return out


def bartlett_sphericity(X: pd.DataFrame) -> tuple[float, float, float]:
    """Test de sphéricité de Bartlett : (chi2, p-value, déterminant de la matrice de corrélation)."""
    n, p = X.shape
    R = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    det = float(np.linalg.det(R))
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    dof = p * (p - 1) / 2
    return float(chi2), float(stats.chi2.sf(chi2, dof)), det


def kmo(X: pd.DataFrame) -> tuple[pd.Series, float]:
    """Indice de Kaiser-Meyer-Olkin : (KMO par variable, KMO global)."""
    R = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    P = np.linalg.inv(R)
    d = np.sqrt(np.outer(np.diag(P), np.diag(P)))
    partial = -P / d
    np.fill_diagonal(R, 0)
    np.fill_diagonal(partial, 0)
    r2, p2 = (R**2).sum(axis=0), (partial**2).sum(axis=0)
    per_var = pd.Series(r2 / (r2 + p2), index=X.columns)
    overall = float((R**2).sum() / ((R**2).sum() + (partial**2).sum()))
    return per_var, overall


def run_pca(X: pd.DataFrame):
    """ACP sur variables standardisées. Retourne (pca, scores, loadings, tableau des valeurs propres)."""
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    scores = pca.transform(Xs)
    names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=X.columns, columns=names)
    table = pd.DataFrame(
        {
            "Composante": names,
            "Valeur propre": pca.explained_variance_,
            "Variance expliquée (%)": pca.explained_variance_ratio_ * 100,
            "Variance cumulée (%)": np.cumsum(pca.explained_variance_ratio_) * 100,
        }
    )
    return pca, scores, loadings, table
