import numpy as np
import pandas as pd
import streamlit as st

import config
from utils import fcfa, load_pipeline, sidebar_footer

st.set_page_config(page_title="Prix des téléphones", page_icon="📱", layout="wide")

# MAE du modèle sur le jeu de test (en FCFA), utilisée pour donner une fourchette indicative
MAE_FCFA = 129_511.53

st.title("📱 Estimer le prix d'un téléphone à Dakar")
st.caption("Régression · Données collectées par web scraping · scikit-learn")

st.markdown(
    """
La base de données de ce projet provient d'un **web scraping d'une plateforme e-commerce de Dakar** :
environ 3 250 annonces de téléphones et tablettes, avec leur prix en FCFA, la marque, l'état, le quartier
du vendeur et les caractéristiques techniques (écran, RAM, stockage).

Des données scrapées sont brutes et bruitées, donc une bonne partie du travail a été le **nettoyage** :
prix irréalistes repérés (jusqu'à 1 FCFA), valeurs aberrantes analysées avec les boxplots et la méthode
de l'IQR, adresses normalisées (accents, ponctuation) et regroupement des quartiers et marques rares dans
une modalité « autre », directement intégrés au pipeline du modèle.
"""
)

# --------------------------------------------------------------------------- #
# Formulaire
# --------------------------------------------------------------------------- #
st.markdown("## Tester le modèle")

BRANDS = {"iPhone": "iPhone", "Samsung": "Samsung", "Xiaomi": "Xiaomi", "ZTE": "ZTE", "iPad": "iPad",
          "Autre marque": "other"}
DISTRICTS = {"Colobane": "colobane", "Liberté 5": "liberte 5", "Médina": "medina", "Ngor": "ngor",
             "Nord Foire": "nord foire", "Ouakam": "ouakam", "Plateau": "plateau", "Autre quartier": "other"}
STATES = ["Neuf", "D'occasion", "Réconditionné", "Venant"]

with st.form("phone_form"):
    c1, c2, c3 = st.columns(3)
    brand_label = c1.selectbox("Marque", list(BRANDS))
    state = c2.selectbox("État", STATES)
    district_label = c3.selectbox("Quartier du vendeur", list(DISTRICTS))

    c1, c2, c3 = st.columns(3)
    screen = c1.number_input("Taille de l'écran (pouces)", min_value=3.0, max_value=14.0, value=6.5, step=0.1)
    ram = c2.selectbox("RAM (Go)", [1, 2, 3, 4, 6, 8, 12, 16], index=3)
    storage = c3.selectbox("Stockage (Go)", [16, 32, 64, 128, 256, 512, 1024], index=3)

    submitted = st.form_submit_button("Estimer le prix", type="primary")

if submitted:
    row = pd.DataFrame(
        [{
            "adresse": DISTRICTS[district_label], "marque": BRANDS[brand_label], "dim_ecr": screen,
            "ram": ram, "stockage": storage, "etat": state,
        }]
    )
    try:
        model = load_pipeline("model_telephone.pkl")
        price = float(np.expm1(model.predict(row)[0]))  # le modèle prédit log1p(prix)
    except Exception as exc:
        st.error(f"Le modèle n'a pas pu être exécuté : {exc}")
        st.stop()

    low, high = max(price - MAE_FCFA, 0), price + MAE_FCFA
    st.metric("Prix estimé", fcfa(price))
    st.caption(f"Fourchette indicative (± l'erreur moyenne du modèle) : {fcfa(low)} – {fcfa(high)}")

# --------------------------------------------------------------------------- #
# Performances
# --------------------------------------------------------------------------- #
st.markdown("## Performances")
st.markdown(
    "Trois modèles linéaires ont été comparés (réglage par validation croisée, 5 plis). Le prix est "
    "modélisé sur une échelle logarithmique (`log1p`) car il est très asymétrique, puis les erreurs sont "
    "recalculées en FCFA pour rester lisibles."
)
perf = pd.DataFrame(
    {
        "Modèle": ["Régression linéaire (retenue)", "Lasso", "Ridge"],
        "R²": [0.2366, 0.2074, 0.2065],
        "RMSE (FCFA)": [185_354, 188_856, 188_971],
        "MAE (FCFA)": [129_512, 131_954, 131_761],
    }
)
st.dataframe(perf, hide_index=True)

st.markdown("## Limites : ce que ce modèle ne sait pas faire")
st.markdown(
    f"""
Soyons clairs : avec un R² d'environ **0,24**, le modèle n'explique qu'une partie des variations de
prix, et l'erreur moyenne est d'environ **{fcfa(MAE_FCFA)}**. Il donne un **ordre de grandeur**, pas un prix précis.

Pistes pour l'améliorer :
- ajouter le **modèle exact** du téléphone (aujourd'hui seule la marque est connue, alors qu'un iPhone 8 et
  un iPhone 15 n'ont pas du tout le même prix) ;
- enrichir avec l'année de sortie ou l'état de la batterie ;
- essayer des modèles non linéaires (forêts aléatoires, gradient boosting) une fois ces variables ajoutées.
"""
)

sidebar_footer(config.REPO_URL)
