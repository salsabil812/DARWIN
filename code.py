# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
# Importation des bibliothèques nécessaires pour l'apprentissage
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Importation des bibliothèques nécessaires pour l'évaluation
from sklearn.metrics import confusion_matrix, f1_score

# 1. Importation du dataset
df = pd.read_csv('C:/Users/Salsabil/Desktop/Darwin/data.csv')

# 2. Vérification des données du dataset
# Afficher les premières lignes pour vérifier l'importation
print(df.head())

# Afficher des informations sur le dataset telles que les types de données et les valeurs manquantes, df.info() retourne none
print(df.info())


# 3. Répartition des classes
# Afficher le nombre d'instances pour chaque classe dans la colonne 'class'
print(df['class'].value_counts())

# 4. Encoder les variables qualitatives (non numériques: de type object dans python)
# Identifier les colonnes catégorielles= colonnes qualitatives (changer selon votre dataset)
categorical_columns = df.select_dtypes(include=['object']).columns

#!!!!!!categorical_columns = categorical_columns.drop('class')  # Exclure la colonne 'class' des colonnes à encoder car Knn demande que la 

# Appliquer le LabelEncoder à chaque colonne catégorielle
label_encoders = {} # Crée un dictionnaire vide appelé 
for column in categorical_columns:
    label_encoders[column] = LabelEncoder() # Pour chaque colonne catégorielle, crée un nouvel objet LabelEncoder() et l'assigne à la clé correspondante dans le dictionnaire (nom de la colonne)
    df[column] = label_encoders[column].fit_transform(df[column]) # Cette méthode fit_transform prend la colonne de données catégorielles et transforme ses valeurs en valeurs numériques en utilisant l'encodeur de labels spécifique à cette colonne.
    #Les valeurs transformées sont ensuite assignées à la colonne correspondante dans le DataFrame d'origine df. Ainsi, les valeurs catégorielles sont remplacées par leurs équivalents numériques dans le DataFrame.


# Vérification de l'encodage ---> type object is converted to int32
print(label_encoders)
print(df.info())

# 5. Feature scaling
# Sélectionner les colonnes numériques
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# Appliquer la standardisation sur les colonnes numériques
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
#Cette opération calcule la moyenne et l'écart type de chaque colonne, puis ajuste les valeurs de manière à ce qu'elles aient une moyenne de 0 et un écart type de 1.
#Les valeurs standardisées sont ensuite assignées aux colonnes correspondantes dans le DataFrame df, remplaçant ainsi les valeurs d'origine.

# 6. Partitionnement du dataset
# Séparer les variables indépendantes (X) de la variable cible (y)
X = df.drop('class', axis=1)  # Supprimer la colonne 'class' des variables indépendantes
y = df['class']  # Définir la variable cible
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) #NB!!!!:  il est crucial de s'assurer que cette variable cible est correctement formatée avant de procéder au fractionnement des données et à l'entraînement du modèle.
print(df.info())
# 7. Création de jeu de données d'apprentissage et de test
# Diviser les données en jeu de données d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)


# Afficher les dimensions des jeux de données pour vérifier les tailles
print("Dimensions du jeu de données d'apprentissage :", X_train.shape)
print("Dimensions du jeu de données de test :", X_test.shape)


# 8. Apprentissage avec K-Nearest Neighbors (KNN)
# Création de l'instance du modèle KNN avec k=5
knn = KNeighborsClassifier(n_neighbors=5)
# Entraînement du modèle KNN sur les données d'apprentissage
knn.fit(X_train, y_train)
# Afficher la performance du modèle sur le jeu de test
print("Score du KNN sur le jeu de test :", knn.score(X_test, y_test))

# 9. Apprentissage avec Multi Layer Perceptron (MLP)
# Création de l'instance du modèle MLP
mlp = MLPClassifier(random_state=42)
# Entraînement du modèle MLP sur les données d'apprentissage
mlp.fit(X_train, y_train)
# Afficher la performance du modèle sur le jeu de test
print("Score du MLP sur le jeu de test :", mlp.score(X_test, y_test))

# 10. Apprentissage avec Support Vector Machine (SVM) et différents noyaux
# Configuration des paramètres pour la validation croisée sur les différents noyaux
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],    
    
}
# Création de l'instance du modèle SVM
svm = SVC()
# Création de l'instance de GridSearchCV pour trouver les meilleurs paramètres
svm_cv = GridSearchCV(svm, param_grid, cv=5)
# Entraînement du modèle SVM avec validation croisée
svm_cv.fit(X_train, y_train)
# Afficher les meilleurs paramètres et la performance du modèle sur le jeu de test
print("Meilleurs paramètres pour SVM :", svm_cv.best_params_)
print("Score du SVM sur le jeu de test :", svm_cv.score(X_test, y_test))


# Fonction pour évaluer un modèle
def evaluate_model(model, X_test, y_test):
    # Prédiction sur le jeu de test
    y_pred = model.predict(X_test)
    
    # Calcul de la matrice de confusion
    confusion = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :\n", confusion)
    
    # Calcul du score F1
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Score F1 :", f1)

# Évaluation de la performance des modèles
print("\nÉvaluation de KNN:")
evaluate_model(knn, X_test, y_test)

print("\nÉvaluation de MLP:")
evaluate_model(mlp, X_test, y_test)

print("\nÉvaluation de SVM:")
evaluate_model(svm_cv, X_test, y_test)
