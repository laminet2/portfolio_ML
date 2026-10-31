import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import skew

import config
from analysis import association_tests, bartlett_sphericity, kmo, load_adult, run_pca
from utils import sidebar_footer

st.set_page_config(page_title="Exploration et ACP", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📊 Exploration et ACP : qui gagne plus de 50 000 $ ?")
st.caption("Statistiques descriptives · Tests d'hypothèses · Analyse en composantes principales")

st.markdown(
    """
Ce projet part du recensement américain de 1994 (*Adult Census Income*) : 14 variables
démographiques et professionnelles décrivent chaque individu, et la cible indique si son revenu annuel dépasse
**50 000 $**. L'étude se déroule en deux temps : d'abord **comprendre et nettoyer** les données, puis se
demander si une **ACP** peut en résumer l'information, et la réponse est plus nuancée qu'on pourrait le croire.
"""
)


@st.cache_data(show_spinner="Chargement des données…")
def get_data():
    return load_adult()


loaded = get_data()

df, n_dup = loaded
num_cols = list(df.select_dtypes(include="number").columns)
cat_cols = list(df.select_dtypes(include="object").columns.drop("income"))
classes = sorted(df["income"].unique())  # ['<=50K', '>50K']
PALETTE = {classes[0]: "#4C78A8", classes[1]: "#E45756"}

# --------------------------------------------------------------------------- #
# 1. Un jeu de données mixte
# --------------------------------------------------------------------------- #
st.markdown("## 1. Un jeu de données mixte")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Individus", f"{len(df):,}".replace(",", " "))
c2.metric("Variables numériques", len(num_cols))
c3.metric("Variables catégorielles", len(cat_cols))
c4.metric("Part des revenus > 50 000 $", f"{(df['income'] == classes[1]).mean():.1%}")
st.markdown(
    "Le problème est une **classification binaire** déséquilibrée : une minorité d'individus dépasse le seuil. "
    "Les variables mélangent numérique et catégoriel, ce qui conditionnera les choix de méthode plus loin."
)

# --------------------------------------------------------------------------- #
# 2. Nettoyer avant d'analyser
# --------------------------------------------------------------------------- #
st.markdown("## 2. Nettoyer avant d'analyser")

missing = (df[cat_cols] == "?").sum()
missing = missing[missing > 0]
miss_txt = ", ".join(f"**{k}** ({v:,})".replace(",", " ") for k, v in missing.items())

st.markdown(
    f"""
- **Doublons** : {n_dup} lignes strictement identiques ont été supprimées.
- **Valeurs manquantes codées « ? »** : elles se concentrent sur {miss_txt or "aucune variable"}.
  Elles sont conservées comme une modalité à part plutôt qu'imputées.
- **Variable redondante** : `education.num` n'est que l'encodage numérique de `education`, elle est supprimée
  pour ne pas compter deux fois la même information.
- **Asymétrie** : `capital.gain`, `capital.loss` et `fnlwgt` ont des distributions très étirées vers la droite ;
  une transformation `log1p` les rend exploitables.
"""
)

skews = pd.Series({c: skew(df[c]) for c in num_cols}, name="Asymétrie (skewness)").round(2)
with st.expander("Voir l'asymétrie de chaque variable numérique après transformation"):
    st.dataframe(skews.to_frame())
    st.markdown(
        "`capital.gain` et `capital.loss` restent fortement asymétriques même après `log1p` : ces variables "
        "sont nulles pour la grande majorité des individus, et aucune transformation ne corrige complètement cela."
    )

# --------------------------------------------------------------------------- #
# 3. Qui gagne plus de 50 000 $ ?
# --------------------------------------------------------------------------- #
st.markdown("## 3. Qui gagne plus de 50 000 $ ?")
st.markdown("Explore la relation entre chaque variable et le revenu :")

var = st.selectbox("Variable à examiner", num_cols + cat_cols)
fig, ax = plt.subplots(figsize=(10, 4))
if var in num_cols:
    sns.boxplot(data=df, x="income", y=var, hue="income", palette=PALETTE, legend=False, ax=ax)
    ax.set_xlabel("Tranche de revenu")
else:
    order = df[var].value_counts().index
    sns.countplot(data=df, x=var, hue="income", order=order, palette=PALETTE, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.set_xlabel("")
    ax.set_ylabel("Effectif")
ax.set_title(f"{var} selon le revenu", fontweight="bold")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

tests = association_tests(df)
st.markdown(
    "Pour aller au-delà des graphiques, chaque variable est testée statistiquement contre le revenu : "
    "**Mann-Whitney** pour les variables numériques (sans hypothèse de normalité, justifiée ici par les tests "
    "de normalité rejetés), **chi-deux** pour les catégorielles."
)
shown = tests.copy()
shown["p-value"] = shown["p-value"].map(lambda p: "< 1e-300" if p == 0 else f"{p:.2e}")
st.dataframe(shown.round({"Statistique": 1}), hide_index=True)

top = tests[tests["Type"] == "catégorielle"].nlargest(3, "Statistique")["Variable"].tolist()
not_sig = tests.loc[tests["p-value"] >= 0.05, "Variable"].tolist()
st.markdown(
    f"""
**Ce que disent les tests :**
- Les variables catégorielles les plus liées au revenu sont **{', '.join(f'`{v}`' for v in top)}**
  (statuts familial et niveau d'études en tête).
- {'La seule variable sans lien significatif avec le revenu est ' + ', '.join(f'`{v}`' for v in not_sig) + ' (poids statistique du recensement, sans signification socio-économique).' if not_sig else 'Toutes les variables ont un lien significatif avec le revenu.'}
- Avec plus de 30 000 individus, presque tout devient « significatif » : la p-value dit qu'un lien existe,
  pas qu'il est fort. C'est la taille de la statistique (chi-deux) qui permet de hiérarchiser.
"""
)

# --------------------------------------------------------------------------- #
# 4. L'ACP a-t-elle un intérêt ici ?
# --------------------------------------------------------------------------- #
st.markdown("## 4. L'ACP a-t-elle un intérêt ici ?")
st.markdown(
    "L'ACP ne s'applique qu'aux variables numériques. Elle n'a de sens que si celles-ci sont **corrélées entre "
    "elles** : c'est cette redondance qu'elle compresse. Avant de la lancer, on vérifie donc si les données s'y prêtent."
)

X = df[num_cols]
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Corrélations entre variables numériques")
fig.tight_layout()
col_a, col_b = st.columns([3, 2])
with col_a:
    st.pyplot(fig)
plt.close(fig)

chi2, p_bart, det = bartlett_sphericity(X)
kmo_vars, kmo_all = kmo(X)
with col_b:
    st.metric("Déterminant de la matrice de corrélation", f"{det:.3f}")
    st.metric("Indice KMO global", f"{kmo_all:.3f}")
    st.metric("Test de sphéricité de Bartlett (χ²)", f"{chi2:,.0f}".replace(",", " "))

st.markdown(
    f"""
Les trois indicateurs vont dans le même sens :
- un **déterminant proche de 1** ({det:.2f}) signifie que la matrice de corrélation est presque la matrice identité,
  donc que les variables sont quasiment indépendantes ;
- un **KMO de {kmo_all:.2f}** est à la limite basse (on cherche au moins 0,6) ;
- le test de Bartlett est certes significatif, mais avec plus de 30 000 observations il détecte la moindre
  corrélation, même négligeable.

**Conclusion : réaliser une ACP sur ces variables n'apportera presque aucun gain** de réduction de dimension.
On la déroule malgré tout pour le vérifier.
"""
)

with st.expander("KMO par variable"):
    st.bar_chart(kmo_vars.rename("KMO"))

# --------------------------------------------------------------------------- #
# 5. Ce que l'ACP montre quand même
# --------------------------------------------------------------------------- #
st.markdown("## 5. Ce que l'ACP montre quand même")
pca, scores, loadings, table = run_pca(X)
n80 = int(np.argmax(table["Variance cumulée (%)"].to_numpy() >= 80) + 1)

st.markdown("### Combien d'axes retenir ?")
col_a, col_b = st.columns([3, 2])
with col_a:
    fig, ax = plt.subplots(figsize=(7, 4))
    comps = np.arange(1, len(table) + 1)
    ax.plot(comps, table["Variance expliquée (%)"], "o-", label="Par composante")
    ax.plot(comps, table["Variance cumulée (%)"], "o--", color="orange", label="Cumulée")
    ax.axhline(80, color="grey", ls=":", lw=1)
    ax.set_xticks(comps)
    ax.set_xlabel("Composante principale")
    ax.set_ylabel("Variance expliquée (%)")
    ax.set_title("Éboulis des valeurs propres")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with col_b:
    st.dataframe(table.round(2), hide_index=True)

st.markdown(
    f"""
Les composantes apportent toutes une part comparable d'information : il faut conserver **{n80} axes sur
{len(table)}** pour atteindre 80 % de la variance, ce qui annule l'intérêt de la réduction de dimension.
"""
)

st.markdown("### Que représente chaque axe ?")
fig, ax = plt.subplots(figsize=(7, 3.6))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Poids des variables sur chaque composante (loadings)")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

lines = []
for pc in loadings.columns:
    v = loadings[pc].abs().idxmax()
    lines.append(f"`{pc}` ← **{v}** ({loadings.loc[v, pc]:+.2f})")
st.markdown(
    "Variable dominante de chaque axe : " + " · ".join(lines) + ".\n\n"
    "À l'exception de `PC1`, qui regroupe partiellement l'âge, le temps de travail et les gains en capital, "
    "chaque axe est porté par **une seule variable isolée** : les variables ne forment pas de sous-groupes "
    "cohérents. L'ACP ne fait ici que redistribuer les variables d'origine, sans structure de synthèse."
)

st.markdown("### Explorer les plans factoriels")
pcs = list(loadings.columns)
c1, c2, c3 = st.columns(3)
x_pc = c1.selectbox("Axe horizontal", pcs, index=0)
y_pc = c2.selectbox("Axe vertical", pcs, index=1)
show_vars = c3.checkbox("Superposer les variables (biplot)", value=True)

if x_pc == y_pc:
    st.info("Choisis deux axes différents.")
else:
    ix, iy = pcs.index(x_pc), pcs.index(y_pc)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(6000, len(df)), replace=False)  # sous-échantillon pour la lisibilité
    plot_df = pd.DataFrame({"x": scores[idx, ix], "y": scores[idx, iy], "income": df["income"].to_numpy()[idx]})

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="income", palette=PALETTE, alpha=0.3, s=18, ax=ax)
    if show_vars:
        scale = 3.5
        for v in loadings.index:
            lx, ly = loadings.loc[v, x_pc] * scale, loadings.loc[v, y_pc] * scale
            ax.arrow(0, 0, lx, ly, color="darkred", head_width=0.12, length_includes_head=True)
            ax.text(lx * 1.15, ly * 1.15, v, color="darkred", ha="center", va="center", fontweight="bold")
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    vr = table.set_index("Composante")["Variance expliquée (%)"]
    ax.set_xlabel(f"{x_pc} ({vr[x_pc]:.1f} %)")
    ax.set_ylabel(f"{y_pc} ({vr[y_pc]:.1f} %)")
    ax.set_title("Individus dans le plan factoriel (6 000 individus tirés au hasard)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Les deux classes de revenu se chevauchent largement : ces axes ne séparent pas les individus selon leur revenu."
    )

# --------------------------------------------------------------------------- #
# 6. Conclusion
# --------------------------------------------------------------------------- #
st.markdown("## 6. Conclusion")
st.markdown(
    f"""
- **Le revenu s'explique surtout par des variables catégorielles** (situation familiale, niveau d'études,
  profession), pas par les cinq variables numériques.
- **L'ACP n'est pas adaptée** : déterminant proche de 1, KMO de {kmo_all:.2f} et {n80} axes nécessaires sur
  {len(table)}, soit aucune réduction de dimension utile.
- Savoir **renoncer à une méthode** après avoir vérifié ses conditions d'application fait partie du travail :
  pour ce jeu de données, la suite logique serait une méthode qui traite variables numériques et
  catégorielles ensemble (analyse factorielle de données mixtes) ou directement un modèle de classification.
"""
)

sidebar_footer(config.REPO_URL)
