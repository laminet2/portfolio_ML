import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from utils import ASSETS_DIR, load_cnn, sidebar_footer

st.set_page_config(page_title="Paludisme · CNN", page_icon="🔬", layout="wide")

st.title("🔬 Diagnostic du paludisme par CNN")
st.caption("Deep Learning · Classification binaire d'images · Keras / TensorFlow")

st.markdown(
    """
Le paludisme se diagnostique en observant au microscope des cellules sanguines : une cellule est
**parasitée** lorsqu'elle contient le parasite *Plasmodium*. Ici, un réseau de neurones convolutif
apprend à faire cette distinction à partir de l'image d'une seule cellule.

Les données viennent du jeu public de la *National Library of Medicine* (NIH) : environ 27 500 images
de cellules, réparties à parts égales entre *Parasitized* et *Uninfected* (80 % pour l'entraînement,
20 % pour le test).
"""
)

# --------------------------------------------------------------------------- #
# Zone de test
# --------------------------------------------------------------------------- #
st.markdown("## Tester le modèle")
st.markdown(
    "Télécharge l'image d'un **patient sain** et l'image d'un **patient malade**, "
    "puis envoie-les ci-dessous pour faire un test."
)

samples = [
    ("patient_saint.png", "Patient sain", "⬇️ Télécharger l'image d'un patient sain"),
    ("patient_malade.png", "Patient malade", "⬇️ Télécharger l'image d'un patient malade"),
]
cols = st.columns(2)
for col, (filename, caption, label) in zip(cols, samples):
    path = ASSETS_DIR / filename
    with col.container(border=True):
        if path.exists():
            st.image(str(path), caption=caption, width=160)
            st.download_button(label, data=path.read_bytes(), file_name=filename, mime="image/png")
        else:
            st.warning(f"Image manquante : assets/{filename}")

uploaded = st.file_uploader("Envoie une image de cellule (PNG ou JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Impossible de lire ce fichier : envoie une image PNG ou JPG valide.")
        st.stop()

    model = load_cnn()
    size = int(model.input_shape[1])  # 64 pour ce modèle

    # Même prétraitement qu'à l'entraînement : redimensionnement (nearest) + normalisation /255
    batch = np.asarray(image.resize((size, size), Image.NEAREST), dtype="float32")[None, ...] / 255.0
    p_uninfected = float(model.predict(batch, verbose=0)[0][0])  # sigmoïde = P(Uninfected)
    p_parasitized = 1.0 - p_uninfected
    infected = p_uninfected <= 0.5

    left, right = st.columns([1, 2])
    left.image(image, caption="Image envoyée", width=200)
    with right:
        if infected:
            st.error(f"🦠 **Cellule parasitée** — confiance : {p_parasitized:.1%}")
        else:
            st.success(f"✅ **Cellule saine** — confiance : {p_uninfected:.1%}")
        st.progress(p_parasitized, text=f"Probabilité d'être parasitée : {p_parasitized:.1%}")

st.caption(
    "⚠️ Démonstration pédagogique : ce modèle ne remplace en aucun cas un diagnostic médical."
)

# --------------------------------------------------------------------------- #
# Démarche et performances
# --------------------------------------------------------------------------- #
st.markdown("## La démarche")
st.markdown(
    """
Je suis parti d'un modèle très simple, puis j'ai amélioré les résultats étape par étape :

1. **CNN de base** : une couche de convolution, un point de départ de référence.
2. **Data augmentation** : des transformations aléatoires des images d'entraînement, pour que le
   modèle généralise mieux au lieu d'apprendre les images par cœur.
3. **Deux couches de convolution + augmentation** : un réseau un peu plus profond, c'est le
   **modèle déployé** dans cette page.
4. **Transfer learning (VGG16)** : un réseau pré-entraîné sur ImageNet, adapté à la tâche.
"""
)

perf = pd.DataFrame(
    {
        "Modèle": [
            "CNN de base",
            "CNN + data augmentation",
            "CNN 2 convolutions + augmentation (déployé)",
            "Transfer learning VGG16",
        ],
        "Précision sur le jeu de test (%)": [88.06, 94.32, 96.05, 92.80],
    }
)
c1, c2 = st.columns([3, 2])
c1.dataframe(perf, hide_index=True)
c2.bar_chart(perf.set_index("Modèle"), horizontal=True)

st.markdown(
    """
**À retenir :** l'augmentation de données apporte le plus gros gain (+6 points). Le réseau à deux
convolutions dépasse même le transfer learning avec VGG16 (96,05 % contre 92,80 %) : un modèle
pré-entraîné n'est donc pas toujours le meilleur choix.
"""
)

sidebar_footer(config.REPO_URL)
