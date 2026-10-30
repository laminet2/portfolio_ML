# Portfolio Machine Learning (Streamlit)

Application Streamlit qui présente quatre projets de Machine Learning, chacun utilisable directement.

| Page | Projet | Modèle |
|------|--------|--------|
| 🔬 Paludisme CNN | Détection de cellules sanguines parasitées | CNN Keras (`models/CnnModel.h5`) |
| 🏦 Marketing bancaire | Probabilité de souscription d'un client | Pipeline XGBoost (`models/bank_model_pipeline.pkl`) |
| 📱 Prix des téléphones | Estimation du prix à Dakar (données scrapées) | Régression linéaire (`models/model_telephone.pkl`) |
| 📊 Exploration et ACP | Recensement américain : EDA, tests, ACP | Calculs en direct |

## Lancer en local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Fichier de données à ajouter

La page « Exploration et ACP » a besoin du jeu de données (non inclus) :
place **`adult.csv`** (Kaggle : *Adult Census Income*, le même que dans le notebook) dans le dossier `data/`.
Le nettoyage du notebook est appliqué automatiquement. Un fichier déjà nettoyé nommé `cleaned_df.csv` est aussi accepté.

## Structure

```
app.py                  page d'accueil
config.py               nom, liens (GitHub, LinkedIn) : à personnaliser
utils.py                transformers personnalisés requis par les .pkl + chargement des modèles
analysis.py             KMO, Bartlett, ACP, tests statistiques
pages/                  une page par projet
models/  assets/  data/
```


