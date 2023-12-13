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

Les "epochs" (époques en français) sont des itérations complètes à travers l'ensemble de données d'entraînement lors de l'apprentissage d'un modèle machine learning. Une époque est atteinte une fois que chaque exemple d'entraînement a été présenté au modèle une fois.

Lorsqu'un modèle est entraîné, il ajuste ses poids et biais pour minimiser la fonction de perte qui mesure la différence entre les prédictions du modèle et les vraies valeurs cibles. Les epochs sont utilisées pour décrire le nombre de fois que l'algorithme d'optimisation traverse l'ensemble complet d'entraînement.

Typiquement, un modèle est entraîné sur plusieurs epochs jusqu'à ce que la performance sur un ensemble de validation cesse de s'améliorer, ou jusqu'à ce qu'un nombre d'époques prédéfini soit atteint.

Il est courant d'ajuster le nombre d'epochs en fonction de la complexité du modèle, de la taille de l'ensemble de données, et du problème spécifique que l'on cherche à résoudre. Trop d'epochs peuvent conduire à un surajustement (overfitting), tandis que trop peu peuvent entraîner un sous-ajustement (underfitting).


----

# Rapport sur la Classification d'Images avec les Réseaux de Neurones - CIFAR-10**

---

### 1. Introduction

Le but de ce travail pratique est de mettre en œuvre des réseaux de neurones pour classer des images à l'aide du dataset CIFAR-10. Nous avons utilisé TensorFlow et Keras pour construire, entraîner et évaluer différents modèles de classification.

### 2. Dataset CIFAR-10

Le dataset CIFAR-10 contient 60 000 images de 32x32 pixels en couleur réparties en 10 classes. La division est de 50 000 images pour l'entraînement et 10 000 pour les tests. L'objectif est de classifier ces images dans les catégories correspondantes.

```python
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

### 3. Exploration du Dataset

Nous avons exploré les données pour comprendre leur structure, la taille, le type, le nombre de classes, et le nombre d'images par classe.

### 4. Prétraitement des Données

Nous avons normalisé les données en les divisant par 255 et converti les étiquettes en encodage one-hot.

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
```

### 5. Réseau de Neurones Dense

Nous avons construit un modèle de réseau de neurones dense simple avec une couche d'entrée, une couche cachée de 512 neurones activés par ReLU, et une couche de sortie avec 10 neurones correspondant aux classes.

```python
model = models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

### 6. Réseau de Neurones Convolutionnels (CNN)

Nous avons construit un modèle CNN plus avancé avec des couches de convolution, de pooling, de dropout, de batch normalization, et une fonction d'activation softmax pour la classification.

```python
model_cnn = models.Sequential()
# Couches de convolution, pooling, dropout, et batch normalization
# ...
model_cnn.add(layers.Dense(10, activation='softmax'))
```

### 7. Entraînement et Évaluation

Nous avons entraîné les modèles sur les données d'entraînement et évalué leurs performances sur les données de test.

```python
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
```

### 8. Améliorations

Nous avons amélioré le modèle CNN en utilisant des techniques telles que l'augmentation de données, le dropout, et la batch normalization.

### 9. Comparaison des Modèles

Nous avons comparé les performances des modèles dense et CNN, notant la supériorité du CNN en termes de capacité à extraire des caractéristiques spatiales.

### 10. Conclusion

Ce travail pratique a permis de mettre en pratique les concepts de réseaux de neurones sur le dataset CIFAR-10. L'expérience a montré l'importance des architectures CNN pour des tâches complexes de classification d'images. Les résultats obtenus montrent l'efficacité des améliorations apportées au modèle CNN.

--- 


### Explications sur le premier code (modèle Dense) :

1. **`num_classes = len(set(y_train.flatten()))`** : Cette ligne de code détermine le nombre de classes dans le jeu de données en comptant le nombre d'éléments uniques dans les étiquettes d'entraînement (y_train) une fois qu'elles ont été aplaties (flatten). Cela est nécessaire pour spécifier le nombre de neurones dans la couche de sortie du modèle, car chaque neurone représentera une classe différente.

2. **One-Hot Encoding** :
    - **`to_categorical(y_train, num_classes)`** : La fonction `to_categorical` convertit les étiquettes (labels) en un format appelé "one-hot encoding". Cela signifie que chaque étiquette est représentée par un vecteur binaire où un seul élément est à 1, indiquant la classe à laquelle l'image appartient. Cette représentation est souvent utilisée dans les problèmes de classification multiclasse. La variable `num_classes` est utilisée ici pour spécifier la dimension du vecteur.

3. **Réseau Dense** :
    - **`model.add(layers.Flatten(input_shape=(32, 32, 3)))`** : Cette ligne ajoute une couche de "flatten" (aplatissement) au modèle. Cela transforme l'entrée, qui est une image 3D (32x32 pixels, 3 canaux pour les couleurs RGB), en un vecteur 1D. Cette couche est nécessaire avant d'ajouter des couches totalement connectées (Dense).
    
    - **`model.add(layers.Dense(512, activation='relu'))`** : Ajoute une couche dense avec 512 neurones et une fonction d'activation "relu" (Rectified Linear Unit). La fonction relu est couramment utilisée dans les couches cachées des réseaux de neurones en raison de sa simplicité et de son efficacité dans la gestion des problèmes de disparition du gradient.

    - **`model.add(layers.Dense(10, activation='relu'))`** : Ajoute une couche dense de sortie avec 10 neurones (correspondant au nombre de classes dans CIFAR-10) et une fonction d'activation "relu". Cependant, pour la couche de sortie, il serait plus approprié d'utiliser une fonction d'activation softmax pour obtenir des probabilités pour chaque classe.

4. **`model.summary()`** : Cette ligne imprime un résumé du modèle, montrant la structure du réseau, le nombre de paramètres, etc.

5. **`model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`** : Configure le processus d'apprentissage du modèle en spécifiant l'optimiseur (adam), la fonction de perte (categorical_crossentropy, car c'est un problème de classification multiclasses), et les métriques à surveiller (précision ici).

6. **`model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))`** : Entraîne le modèle sur les données d'entraînement (`x_train`, `y_train`) avec 20 epochs, une taille de lot de 32, en utilisant les données de validation (`x_test`, `y_test`) pour évaluer les performances pendant l'entraînement.

### Explications sur le deuxième code (modèle CNN) :

1. **Réseau Convolutionnel (CNN)** :
    - **`model_cnn = models.Sequential()`** : Crée un modèle séquentiel, une pile linéaire de couches.
    
    - **`model_cnn.add(layers.BatchNormalization())`** : Ajoute une couche de normalisation des lots. Elle normalise l'activation de chaque neurone pour accélérer l'apprentissage.
    
    - **`model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))`** : Ajoute une couche de convolution 2D avec 32 filtres de taille (3, 3), une fonction d'activation relu et une forme d'entrée de (32, 32, 3). Cela capture les caractéristiques spatiales de l'image.
    
    - **`model_cnn.add(layers.Dropout(0.05))`** : Ajoute une couche de "dropout" pour éviter le surajustement en désactivant aléatoirement certains neurones pendant l'entraînement.
    
    - **`model_cnn.add(layers.MaxPooling2D((2, 2)))`** : Ajoute une couche de pooling pour réduire la dimension spatiale de la représentation.

2. **Compilation et Entraînement** :
    - **`model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`** : Configure le modèle pour l'apprentissage avec l'optimiseur Adam, la fonction de perte categorical_crossentropy, et la métrique accuracy.
    
    - **`model_cnn.fit(x_train, y_train, batch_size=16, epochs=13, validation_data=(x_test, y_test), callbacks=callbacks_list)`** : Entraîne le modèle sur les données d'entraînement avec un batch size de 16, pendant 13 epochs, en utilisant les données de validation pour évaluer les performances. Les callbacks sont utilisés pour enregistrer le meilleur modèle en fonction de la précision sur les données de validation.



**One-Hot Encoding :**

Le One-Hot Encoding est une technique utilisée en apprentissage machine pour représenter catégoriquement des variables. Elle est particulièrement utilisée pour encoder les étiquettes dans les tâches de classification. 

**Exemple simple :**

Supposons que nous ayons un ensemble de données contenant des fruits avec des étiquettes comme suit : pomme, banane, orange. Ces étiquettes sont des catégories, et nous voulons les encoder pour les utiliser dans un modèle d'apprentissage machine.

- **Sans One-Hot Encoding :**
  - Pomme → 1
  - Banane → 2
  - Orange → 3

  Ici, les nombres représentent les catégories, mais ils ne portent pas de signification intrinsèque entre eux. Un modèle pourrait interpréter que l'orange (3) est plus similaire à la banane (2) qu'à la pomme (1), ce qui peut être problématique.

- **Avec One-Hot Encoding :**
  - Pomme → [1, 0, 0]
  - Banane → [0, 1, 0]
  - Orange → [0, 0, 1]

  Avec One-Hot Encoding, chaque catégorie est représentée par un vecteur binaire où une seule position correspond à la catégorie réelle. Cela élimine l'ambiguïté que pourrait introduire une représentation numérique directe.

**Définition simple :**

Le One-Hot Encoding consiste à convertir une variable catégorielle en un vecteur binaire qui indique la présence ou l'absence de chaque catégorie. Chaque catégorie est représentée par une position unique dans le vecteur. Cela facilite l'utilisation de variables catégorielles dans les modèles d'apprentissage machine, car cela évite d'imposer des relations numériques artificielles entre les catégories.

----

Chacune des fonctions et méthodes mentionnées appartient à l'API Keras, une interface haut niveau pour la construction et l'entraînement de modèles d'apprentissage profond. Voici à quoi chaque fonction sert :

1. **`Flatten` :**
   - **Fonction :** La fonction `Flatten` est utilisée pour aplatir les données d'entrée. Elle est souvent utilisée comme première couche d'un modèle séquentiel pour convertir des données 2D (comme des images) en un vecteur 1D, ce qui est nécessaire avant d'ajouter des couches denses.
   - **Exemple :** `model.add(layers.Flatten(input_shape=(32, 32, 3)))`

2. **`Dense` :**
   - **Fonction :** La fonction `Dense` est utilisée pour ajouter des couches de neurones densément connectés. C'est la couche classique que l'on trouve dans de nombreux réseaux de neurones. Chaque neurone dans une couche dense est connecté à tous les neurones de la couche précédente.
   - **Exemple :** `model.add(layers.Dense(512, activation='relu'))`

3. **`compile` :**
   - **Fonction :** La méthode `compile` est utilisée pour configurer le modèle pour l'entraînement. Elle prend en paramètre l'optimiseur, la fonction de perte, et les métriques à surveiller pendant l'entraînement.
   - **Exemple :** `model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`

4. **`fit` :**
   - **Fonction :** La méthode `fit` est utilisée pour entraîner le modèle sur des données d'entraînement. Elle spécifie le nombre d'epochs, la taille du batch, et peut également inclure des données de validation.
   - **Exemple :** `model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))`

5. **`summary` :**
   - **Fonction :** La méthode `summary` est utilisée pour afficher un résumé du modèle, montrant la structure des couches, le nombre de paramètres, et d'autres informations utiles.
   - **Exemple :** `model.summary()`

6. **`save` :**
   - **Fonction :** La méthode `save` est utilisée pour sauvegarder l'ensemble du modèle, y compris son architecture, ses poids, et ses configurations, dans un fichier.
   - **Exemple :** `model.save('model.keras')`

Ces fonctions et méthodes facilitent la création, la configuration, l'entraînement, et la sauvegarde des modèles d'apprentissage profond en utilisant l'API Keras.

-----
