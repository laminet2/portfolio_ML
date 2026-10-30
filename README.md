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


## Structure

```
app.py                  page d'accueil
config.py               nom, liens (fichier de configurations)
utils.py                transformers personnalisés requis par les .pkl + chargement des modèles
analysis.py             KMO, Bartlett, ACP, tests statistiques
pages/                  une page par projet
models/  assets/  data/
```


