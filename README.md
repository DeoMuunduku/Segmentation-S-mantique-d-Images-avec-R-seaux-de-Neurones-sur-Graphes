# Segmentation-S-mantique-d-Images-avec-R-seaux-de-Neurones-sur-Graphes


Ce projet vise à implémenter une solution de segmentation sémantique d'images en utilisant des réseaux de neurones sur graphes, avec une application pratique sur le jeu de données de Cityscapes.
# le code ne pas complet et ce juste une partie fine du projet parceque le travaille est encour d'etre accepter .....
## Auteur

Ce projet a été réalisé par Deo.

## Introduction

La segmentation sémantique d'images est une tâche de vision par ordinateur qui consiste à attribuer une étiquette sémantique à chaque pixel d'une image. Dans ce projet, nous utilisons des réseaux de neurones sur graphes pour réaliser cette tâche.

## Jeu de Données

Le jeu de données utilisé dans ce projet est [Cityscapes](https://www.cityscapes-dataset.com/), qui contient un ensemble d'images de rues urbaines annotées avec des étiquettes sémantiques pour des classes telles que les voitures, les piétons, les routes, etc.

Pour accéder au jeu de données, veuillez vous inscrire sur le site officiel de Cityscapes et télécharger les fichiers nécessaires.

## Architecture du Modèle

Dans ce projet, nous utilisons une architecture inspirée de U-Net pour effectuer la segmentation sémantique. Le modèle est entraîné sur les images annotées du jeu de données Cityscapes.

## Prétraitement des Données

Les images du jeu de données sont prétraitées pour les adapter à la forme d'entrée du modèle, et les annotations sont converties en masques binaires correspondant aux différentes classes.

## Entraînement du Modèle

Le modèle est entraîné sur les données d'entraînement en utilisant l'optimiseur Adam et la perte de cross-entropy catégorielle. Les performances du modèle sont évaluées à l'aide de différentes métriques telles que l'intersection over union (IoU) sur un ensemble de données de validation.

## Exécution du Code

Pour exécuter le code, suivez les instructions suivantes :

1. Assurez-vous d'avoir téléchargé et extrait le jeu de données Cityscapes.
2. Placez les images et les annotations dans les répertoires appropriés (par exemple, 'train/images' et 'train/labels').
3. Exécutez le script principal pour entraîner et évaluer le modèle.

## Remarques

- Ce projet est réalisé à des fins éducatives et peut être étendu ou amélioré selon les besoins.( a la fn de ce projet )
- N'hésitez pas à explorer différentes architectures de modèles, techniques d'augmentation de données, etc., pour améliorer les performances du modèle.

