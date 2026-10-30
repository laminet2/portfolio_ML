import streamlit as st

import config
from utils import sidebar_footer

st.set_page_config(page_title="Portfolio Machine Learning", page_icon="🧠", layout="wide")

# --------------------------------------------------------------------------- #
# En-tête
# --------------------------------------------------------------------------- #
st.title("🧠 Portfolio Machine Learning")
st.subheader(config.NAME)
st.caption(config.TITLE)

st.markdown(
    """
Cette application rassemble quelques-uns de mes projets de Machine Learning. Chacun est **utilisable
directement** : on y teste le modèle entraîné, on y consulte ses performances et on y retrouve
la démarche qui y a mené (nettoyage, choix de modèles, limites).
"""
)

links = [f"[GitHub]({config.GITHUB_PROFILE})"]
if config.LINKEDIN_URL:
    links.append(f"[LinkedIn]({config.LINKEDIN_URL})")
st.markdown("**Me retrouver :** " + " · ".join(links))
st.markdown(f"**Compétences :** {config.SKILLS}")

st.info(f"💻 Retrouvez le code source de ce repo sur [GitHub]({config.REPO_URL}).")

# --------------------------------------------------------------------------- #
# Les projets
# --------------------------------------------------------------------------- #
st.markdown("## Les projets")

projects = [
    {
        "page": "pages/1_Paludisme_CNN.py",
        "icon": "🔬",
        "title": "Diagnostic du paludisme (CNN)",
        "kind": "Deep Learning · Vision par ordinateur",
        "text": "Un réseau de neurones convolutif détecte si une cellule sanguine est parasitée à partir "
        "d'une simple image de frottis.",
    },
    {
        "page": "pages/2_Marketing_Bancaire.py",
        "icon": "🏦",
        "title": "Marketing bancaire",
        "kind": "Classification · XGBoost",
        "text": "Prédire la probabilité qu'un client souscrive suite à une campagne d'appels, avec un "
        "traitement soigné des valeurs « unknown ».",
    },
    {
        "page": "pages/3_Prix_Telephones.py",
        "icon": "📱",
        "title": "Prix des téléphones à Dakar",
        "kind": "Régression · Données scrapées",
        "text": "Estimer le prix d'un téléphone (en FCFA) à partir de sa marque, de son état et de ses "
        "caractéristiques techniques.",
    },
    {
        "page": "pages/4_Exploration_ACP.py",
        "icon": "📊",
        "title": "Exploration et ACP",
        "kind": "Statistiques · Analyse de données",
        "text": "Du recensement américain (revenu > 50 000 $) : analyse exploratoire, tests statistiques "
        "puis ACP, avec la question « l'ACP est-elle pertinente ici ? ».",
    },
]

cols = st.columns(2)
for i, p in enumerate(projects):
    with cols[i % 2].container(border=True):
        st.markdown(f"### {p['icon']} {p['title']}")
        st.caption(p["kind"])
        st.write(p["text"])
        st.page_link(p["page"], label="Ouvrir le projet", icon="➡️")

sidebar_footer(config.REPO_URL)
