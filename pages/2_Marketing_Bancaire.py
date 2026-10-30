import pandas as pd
import streamlit as st

import config
from utils import load_pipeline, sidebar_footer

st.set_page_config(page_title="Marketing bancaire", page_icon="🏦", layout="wide")

st.title("🏦 Marketing bancaire : qui va souscrire ?")
st.caption("Classification binaire · Pipeline scikit-learn · XGBoost")

st.markdown(
    """
Une banque appelle ses clients pour leur proposer un produit. Appeler coûte du temps : l'objectif est
de **prédire la probabilité qu'un client souscrive**, afin de cibler les appels les plus prometteurs.
Le modèle est un pipeline complet (prétraitement + XGBoost) : on lui donne les informations brutes
d'un client et il se charge du reste.
"""
)

# --------------------------------------------------------------------------- #
# Formulaire de prédiction
# --------------------------------------------------------------------------- #
st.markdown("## Tester le modèle")

with st.form("bank_form"):
    st.markdown("**Le client**")
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Âge", 18, 95, 40)
    job = c2.selectbox(
        "Profession",
        ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
         "self-employed", "services", "student", "technician", "unemployed", "unknown"],
        index=0,
    )
    marital = c3.selectbox("Situation familiale", ["married", "single", "divorced", "unknown"])

    c1, c2, c3 = st.columns(3)
    education = c1.selectbox(
        "Niveau d'études",
        ["university.degree", "high.school", "professional.course", "basic.9y", "basic.6y",
         "basic.4y", "illiterate", "unknown"],
    )
    housing = c2.selectbox("Crédit immobilier", ["no", "yes", "unknown"])
    loan = c3.selectbox("Crédit personnel", ["no", "yes", "unknown"])

    st.markdown("**La campagne**")
    c1, c2, c3 = st.columns(3)
    contact = c1.selectbox("Moyen de contact", ["cellular", "telephone"])
    month = c2.selectbox(
        "Mois du dernier contact", ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], index=2
    )
    day_of_week = c3.selectbox("Jour de la semaine", ["mon", "tue", "wed", "thu", "fri"])

    c1, c2, c3 = st.columns(3)
    duration = c1.number_input("Durée de l'appel (secondes)", min_value=0, max_value=5000, value=180, step=10)
    campaign = c2.number_input("Nombre de contacts pendant cette campagne", min_value=1, max_value=60, value=2)
    previous = c3.number_input("Contacts lors de campagnes précédentes", min_value=0, max_value=6, value=0)

    c1, c2 = st.columns(2)
    never_contacted = c1.checkbox("Jamais contacté auparavant", value=True)
    pdays_value = c2.number_input("Jours depuis le dernier contact", min_value=0, max_value=999, value=30,
                                  disabled=never_contacted)
    poutcome = st.selectbox("Résultat de la campagne précédente", ["nonexistent", "failure", "success"])

    submitted = st.form_submit_button("Calculer la probabilité", type="primary")

if submitted:
    row = pd.DataFrame(
        [{
            "age": age, "job": job, "marital": marital, "education": education,
            "housing": housing, "loan": loan, "contact": contact, "month": month,
            "day_of_week": day_of_week, "duration": duration, "campaign": campaign,
            "pdays": 999 if never_contacted else pdays_value,  # 999 = jamais contacté
            "previous": previous, "poutcome": poutcome,
        }]
    )
    try:
        model = load_pipeline("bank_model_pipeline.pkl")
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as exc:  # affichage lisible plutôt qu'une page rouge
        st.error(f"Le modèle n'a pas pu être exécuté : {exc}")
        st.stop()

    st.metric("Probabilité de souscription", f"{proba:.1%}")
    st.progress(proba)
    if proba >= 0.5:
        st.success("Profil **prometteur** : ce client a de bonnes chances de souscrire.")
    else:
        st.info("Profil **peu probable** : à ne contacter qu'en second lieu.")

st.caption(
    "ℹ️ La durée de l'appel n'est connue qu'**après** l'appel : elle sert ici à démontrer le modèle, "
    "mais elle ne serait pas disponible pour décider qui appeler dans un vrai déploiement."
)

# --------------------------------------------------------------------------- #
# Comparaison des modèles
# --------------------------------------------------------------------------- #
st.markdown("## Comparaison des modèles")
st.markdown(
    "Trois modèles d'ensemble ont été optimisés par validation croisée (GridSearchCV, 5 plis) puis "
    "évalués sur 20 % de données mises de côté. Le meilleur selon le F1-score a été retenu."
)

results = pd.DataFrame(
    {
        "Modèle": ["XGBoost (retenu)", "Gradient Boosting", "Random Forest"],
        "Accuracy": [0.8548, 0.8540, 0.8471],
        "Precision": [0.8073, 0.8128, 0.8023],
        "Recall": [0.8351, 0.8233, 0.8179],
        "F1-score": [0.8210, 0.8180, 0.8100],
        "ROC-AUC": [0.9258, 0.9255, 0.9217],
    }
)
st.dataframe(results, hide_index=True)
st.bar_chart(results.set_index("Modèle")[["Accuracy", "F1-score", "ROC-AUC"]])
st.markdown(
    "Les trois modèles sont très proches ; l'écart est faible entre eux, donc le choix d'XGBoost "
    "(50 arbres, profondeur 4) repose surtout sur un léger avantage en F1-score et en recall."
)

# --------------------------------------------------------------------------- #
# Ce que j'ai découvert
# --------------------------------------------------------------------------- #
st.markdown("## Ce que l'analyse des données a révélé")

st.markdown(
    """
**Les « unknown » ne sont pas du hasard.** Plusieurs variables (profession, études, crédits) contiennent
la modalité *unknown*. La tentation est de les traiter comme des valeurs manquantes et de les remplacer
par la modalité la plus fréquente, mais ces absences ne sont pas aléatoires : les imputer aurait créé
des corrélations artificielles. Elles sont donc conservées comme une modalité à part entière.

L'exemple le plus parlant : l'association entre *crédit immobilier* et *crédit personnel* semblait forte
(V de Cramér = 0,71). Une fois les « unknown » retirés, elle tombe à **0,04**. Cette forte association
n'était que le reflet de clients dont les deux informations manquaient en même temps.

**`pdays` est en réalité une variable catégorielle.** Elle vaut 999 pour la grande majorité des clients
(jamais contactés auparavant). Un arbre de décision peu profond a été utilisé pour trouver le seuil de
coupure optimal (513 jours), ce qui sépare 1 069 clients « contactés récemment » des 10 570 autres.

**Variables très asymétriques.** `duration` et `campaign` ont de longues queues à droite : elles sont
transformées par `log1p` avant la standardisation, à l'intérieur du pipeline.
"""
)

sidebar_footer(config.REPO_URL)
