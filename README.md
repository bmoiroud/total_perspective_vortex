# Total Perspective Vortex

## Description

Ce projet a pour but recréer l'algorithme CSP pour faciliter la détection de différents mouvements dans un signal eeg.

Les enregistrements proviennent : https://doi.org/10.13026/C28G6P.

## Installation

``` bash
$> pip install -r ./requirements.txt
```

## Entrainement
```bash
$> py train.py ./datasets/data_train.csv
```

## Prédiction
```bash
$> py predict.py ./datasets/data_predict.csv
```