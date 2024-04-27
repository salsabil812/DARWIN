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

df = pd.read_csv('C:/Users/Salsabil/Desktop/Darwin/data.csv')
print(df.head())
print(df.info())
print(df['class'].value_counts())

categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {} 
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column]) 
    
print(label_encoders)
print(df.info())

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

X = df.drop('class', axis=1) 
y = df['class']  # C'est la variable cible
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) 
print(df.info())

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
print("Dimensions du jeu de données d'apprentissage :", X_train.shape)
print("Dimensions du jeu de données de test :", X_test.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Score du KNN sur le jeu de test :", knn.score(X_test, y_test))

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Score du MLP sur le jeu de test :", mlp.score(X_test, y_test))

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],    
    
}
svm = SVC()
svm_cv = GridSearchCV(svm, param_grid, cv=5)
svm_cv.fit(X_train, y_train)

print("Meilleurs paramètres pour SVM :", svm_cv.best_params_)
print("Score du SVM sur le jeu de test :", svm_cv.score(X_test, y_test))

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)    
    confusion = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :\n", confusion)    
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Score F1 :", f1)

# Évaluation de la performance des modèles
print("\nÉvaluation de KNN:")
evaluate_model(knn, X_test, y_test)

print("\nÉvaluation de MLP:")
evaluate_model(mlp, X_test, y_test)

print("\nÉvaluation de SVM:")
evaluate_model(svm_cv, X_test, y_test)
