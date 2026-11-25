# EEG Spike Explorer

Détection automatique de **pointes intercritiques** et analyse de **connectivité entre électrodes** sur de l'EEG de scalp d'enfants épileptiques, avec **MNE-Python**, un cœur de détection en **Rust** (PyO3) et un visualiseur interactif **WebGPU**.

Projet réalisé en une soirée (novembre 2026) pour montrer une chaîne complète, de l'EDF brut à une interface utilisable par un clinicien, en m'appuyant sur un assistant IA pour aller vite sur les parties interface et Rust.

![Visualiseur : tracé EEG avec pointes détectées, réseau sur le scalp, matrice de connectivité](docs/viewer-dark.png)

## Ce que fait le projet

```
EDF ──► MNE : notch 50 Hz, passe-bande 1-70 Hz, rééchantillonnage 250 Hz, montage bipolaire 10-20
    ──► détection de pointes (Teager-Kaiser + seuil robuste)   ← NumPy ou Rust, résultats identiques
    ──► regroupement multi-canaux en événements
    ──► connectivité : wPLI par bande (MNE-Connectivity), corrélation, co-occurrence des pointes
    ──► graphe NetworkX : force, centralité, canaux « hubs »
    ──► viewer.html autonome (WebGPU, repli Canvas 2D) + CSV/JSON
```

Sur l'enregistrement de démonstration (sujet 01, 5 min de sommeil N3) :

| | |
|---|---|
| Événements détectés | 534 (≈ 107 / min, activité pointe-onde quasi continue) |
| Canaux les plus impliqués | T5-O1, Fp1-F3, C3-P3 (co-occurrence) |
| Détection Rust vs NumPy | ≈ 4 à 7× plus rapide, sorties identiques (testé) |
| Pipeline complet | ≈ 5 s sur un portable |

## Démarrage rapide

```bash
pip install -e .                       # paquet Python (MNE, MNE-Connectivity, NetworkX…)
pip install maturin && (cd rust && maturin build --release && pip install target/wheels/*.whl)   # optionnel : cœur Rust
scripts/download_data.sh 01 01         # 5 min d'EEG, OpenNeuro ds003555 (CC0)
eegspike run data/sub-01_run-01.edf -o results
open results/viewer.html               # Chrome / Edge / Safari 18+ pour WebGPU
```

Sorties dans `results/` : `spike_events.csv`, `spike_candidates.csv`, `graph_hubs.csv`, `connectivity.npz`, `summary.json`, `viewer.html`.

Utilisation en bibliothèque :

```python
import eegspike as es
raw = es.load_eeg("data/sub-01_run-01.edf")
spikes = es.detect_spikes(raw)                 # backend="auto" | "rust" | "numpy"
mats = es.compute_connectivity(raw, spikes.events)
g = es.connectivity_graph(mats["wpli_delta"], raw.ch_names)
```

## Méthode

**Détection** (`src/eegspike/detect.py`, `rust/src/lib.rs`)
1. Passe-bande 3-35 Hz pour garder la composante pointue des pointes.
2. Opérateur d'énergie de Teager-Kaiser `x[t]² − x[t−1]·x[t+1]`, grand quand le signal est à la fois ample et rapide, lissé sur 20 ms.
3. Seuil adaptatif par canal : `médiane + k·MAD/0.6745` (k = 15), robuste aux artefacts.
4. Maxima locaux au-dessus du seuil, période réfractaire de 200 ms.
5. Une pointe a un champ spatial : on garde les détections vues sur ≥ 3 canaux en moins de 80 ms.

Les étapes 2 à 4 existent en NumPy (référence lisible) et en Rust (multithread, sans GIL). Les tests vérifient que les deux donnent exactement les mêmes pointes.

**Connectivité** (`src/eegspike/connectivity.py`)
- **wPLI** (weighted Phase Lag Index) sur des époques de 2 s, bandes delta, thêta, alpha, bêta. Il ignore le couplage à déphasage nul, donc limite l'effet de conduction volumique.
- **Co-occurrence des pointes** (indice de Jaccard) : quels canaux « pointent » ensemble, une approximation de l'étendue de la zone irritative.
- **Corrélation** comme référence simple.
- Graphe NetworkX sur les 15 % de liens les plus forts : force, centralité de degré et d'intermédiarité.

**Visualiseur** (`src/eegspike/viewer_template.html`)
- Un seul fichier HTML, sans serveur ni dépendance, généré par Python avec les données embarquées (int16, ≈ 3,7 Mo pour 5 min × 18 canaux).
- Rendu WebGPU : tout le signal est chargé une fois dans un *storage buffer* ; un shader WGSL dessine chaque canal comme une instance de `line-strip`. Se déplacer dans le tracé ne met à jour que 64 octets d'uniformes. Repli automatique en Canvas 2D si WebGPU est absent.
- Navigation clavier (← →, N/P pour pointe suivante/précédente, +/− pour le gain), frise des événements cliquable, réseau sur le scalp et matrice de connectivité au choix de la mesure.

## Tests

```bash
pip install pytest && pytest -q        # Python : détection sur pointes synthétiques, Rust == NumPy, graphe
cd rust && cargo test                  # Rust : tests unitaires du cœur
```

## Données

[OpenNeuro ds003555](https://openneuro.org/datasets/ds003555) : EEG de sommeil de 30 enfants épileptiques (Hôpital pédiatrique de Zurich), segments N3 de 5 minutes en montage bipolaire, 2000 Hz, licence CC0. Le code lit n'importe quel EDF : il prend le montage « double banane » si les canaux existent, sinon tous les canaux EEG. Il fonctionne donc aussi sur CHB-MIT (PhysioNet).

## Limites et suites

- **Pas un outil clinique.** Le détecteur est volontairement simple et n'a pas été validé contre des annotations d'experts (ce jeu de données annote les HFO, pas les pointes). L'étape suivante serait une évaluation sensibilité / précision sur un jeu annoté, puis la comparaison avec un détecteur appris (CNN 1D en PyTorch).
- Connectivité en espace capteurs bipolaire : utile pour comparer des canaux, mais l'interprétation anatomique demanderait une reconstruction de sources.
- Interface : sélection de canaux, annotation manuelle des pointes et export, intégration dans MNE-Qt-Browser ou une application Qt.

## Auteur

Valentin Goupille, M2 bio-informatique. [valentingoupille.com](https://valentingoupille.com)
