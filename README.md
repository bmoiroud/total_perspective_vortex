# Total Perspective Vortex

## Description

Le but de ce sujet est de créer une interface cerveau machine basée sur des données d’électroencéphalogramme (EEG) à l’aide de machine learning. À partir d’un jeu de données on doit déterminer si la personne dont on a mesuré l’activité cérébrale
pense ou effectue, un mouvement A ou B.

Source des données: https://doi.org/10.13026/C28G6P.

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