# TP4 réseaux de neurones 

### A quoi sert de construire un réseau de neurones dense pour la classification d’images. 
---

**Rapport d'Explications : Construction d'un Réseau de Neurones Dense pour la Classification d'Images**

Dans le cadre de ce projet, nous avons utilisé TensorFlow et Keras pour construire un réseau de neurones dense visant à classifier des images à l'aide du dataset CIFAR-10. Les composants clés utilisés pour la construction du modèle sont les classes `Sequential` et `Dense`.

1. **Sequential :**

   La classe `Sequential` de Keras est un conteneur linéaire séquentiel permettant de créer des modèles de manière itérative. Elle facilite la construction d'un modèle couche par couche, en ajoutant successivement des couches à l'aide de la méthode `add`.

   *Rôle :*
   - **Construction itérative :** `Sequential` permet de construire des modèles de manière séquentielle, en ajoutant des couches les unes après les autres. Cela offre une approche intuitive et simple pour créer des architectures de réseaux de neurones.

   *Utilisation :*
   - Nous avons instancié un modèle `Sequential` comme base pour notre réseau de neurones. En utilisant la méthode `add`, nous avons ajouté différentes couches pour construire l'architecture souhaitée.

2. **Dense :**

   La classe `Dense` est une couche de neurones entièrement connectée. Chaque neurone dans cette couche est connecté à tous les neurones de la couche précédente.

   *Rôle :*
   - **Connectivité complète :** Chaque neurone dans une couche `Dense` est connecté à tous les neurones de la couche précédente, fournissant ainsi une connectivité complète entre les couches.
   - **Apprentissage de représentations :** Les couches `Dense` sont responsables de l'apprentissage de représentations complexes à partir des données d'entrée.

   *Utilisation :*
   - Nous avons utilisé des couches `Dense` pour créer les différentes parties du modèle, notamment la couche d'aplatissement pour convertir les images en vecteurs, une couche intermédiaire avec une fonction d'activation ReLU pour extraire des caractéristiques, et enfin une couche de sortie avec une activation softmax pour la classification.

En résumé, l'utilisation de `Sequential` et `Dense` dans la construction du modèle offre une approche structurée et flexible pour concevoir des réseaux de neurones denses pour la classification d'images. Ces composants facilitent la création, la gestion et la compréhension des architectures de réseaux de neurones pour des tâches variées.


----

## c'est quoi les epochs 