#!/usr/bin/env bash
# Télécharge un segment EEG de 5 min (N3, montage bipolaire) du jeu OpenNeuro ds003555
# "EEG recordings of 30 pediatric patients with epilepsy" (licence CC0).
# Usage : scripts/download_data.sh [sujet=01] [run=01]
set -euo pipefail
SUB=${1:-01}; RUN=${2:-01}
mkdir -p data
URL="https://s3.amazonaws.com/openneuro.org/ds003555/derivatives/sub-${SUB}/ses-01/eeg/sub-${SUB}_ses-01_task-hfo_run-${RUN}_eeg.edf"
echo "→ $URL"
curl -fL -o "data/sub-${SUB}_run-${RUN}.edf" "$URL"
echo "OK : data/sub-${SUB}_run-${RUN}.edf"
