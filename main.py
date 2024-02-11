import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Wczytaj dane z pliku CSV
data = pd.read_csv('Iris.csv')

# Podziel dane na funkcje (cechy) i etykiety (klasy)
X = data.drop('Species', axis=1)
y = data['Species']

# Podziel dane na zbiór treningowy i testowy (np. 70% treningowy, 30% testowy)
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=5)

sns.pairplot(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']], hue='Species')






plt.show()



#3-4-5 zadanie
# Tworzenie modelu K-nearest neighbors (KNN) z liczbą sąsiadów ustawioną na 1
KNN_model = KNeighborsClassifier(n_neighbors=1)
# Trenowanie modelu KNN na zbiorze treningowym
KNN_model.fit(X_train, y_train)
# Przewidywanie etykiet na podstawie danych testowych
y_pred = KNN_model.predict(X_test)
# Obliczenie dokładności klasyfikacji modelu KNN na danych testowych
k_scores = metrics.accuracy_score(y_test, y_pred)
# Wygenerowanie raportu klasyfikacji, który zawiera miary, takie jak precyzja, czułość, F1-score itp.
classification_rep_knn = classification_report(y_test, y_pred)
# Tworzenie macierzy pomyłek, która pokazuje liczbę poprawnych i błędnych klasyfikacji dla każdej klasy
conf_matrix_knn = confusion_matrix(y_test, y_pred)
# Wyświetlenie wyniku dokładności klasyfikacji modelu KNN
print("The Result of acurracy in method KNN = ",k_scores)
# Wyświetlenie raportu klasyfikacji, który dostarcza bardziej szczegółowych informacji o wydajności modelu
print("Classification of the KNN = \n",classification_rep_knn)
# Wyświetlenie macierzy pomyłek, która pomaga zrozumieć, jak model klasyfikuje różne klasy
print("Matrix of the KNN = \n",conf_matrix_knn,"\n")


# Tworzenie modelu SVM z jądrem liniowym i parametrem C ustawionym na 4
SVM_model = svm.SVC(kernel='linear', C=4)
# Trenowanie modelu SVM na zbiorze treningowym
SVM_model.fit(X_train, y_train)
# Przewidywanie etykiet na podstawie danych testowych
y_pred = SVM_model.predict(X_test)
# Obliczenie dokładności klasyfikacji modelu SVM na danych testowych
svm_scores = metrics.accuracy_score(y_test, y_pred)
# Wygenerowanie raportu klasyfikacji, który zawiera miary, takie jak precyzja, czułość, F1-score itp.
classification_rep_svm = classification_report(y_test, y_pred)
# Tworzenie macierzy pomyłek, która pokazuje liczbę poprawnych i błędnych klasyfikacji dla każdej klasy
conf_matrix_svm = confusion_matrix(y_test, y_pred)
# Wyświetlenie wyniku dokładności klasyfikacji modelu SVM
print("Result of the acurracy in method SVM =", svm_scores)
# Wyświetlenie raportu klasyfikacji, który dostarcza bardziej szczegółowych informacji o wydajności modelu
print("Classification of the SVM = \n",classification_rep_svm)
# Wyświetlenie macierzy pomyłek, która pomaga zrozumieć, jak model klasyfikuje różne klasy
print("Matrix of the SVM = \n",conf_matrix_svm,"\n")



# Tworzenie modelu Random Forests z 100 drzewami i ustalonym stanem losowym
RF_model = RandomForestClassifier(n_estimators=100, random_state=5)
# Trenowanie modelu Random Forests na zbiorze treningowym
RF_model.fit(X_train, y_train)
# Przewidywanie etykiet na podstawie danych testowych
y_pred = RF_model.predict(X_test)
# Obliczenie dokładności klasyfikacji modelu Random Forests na danych testowych
RF_scores = metrics.accuracy_score(y_test, y_pred)
# Wygenerowanie raportu klasyfikacji, który zawiera miary, takie jak precyzja, czułość, F1-score itp.
classification_rep_rf = classification_report(y_test, y_pred)
# Tworzenie macierzy pomyłek, która pokazuje liczbę poprawnych i błędnych klasyfikacji dla każdej klasy
conf_matrix_rf = confusion_matrix(y_test, y_pred)
# Wyświetlenie dokładności klasyfikacji modelu Random Forests
print("Result of the acurracy in method RF =", RF_scores)
# Wyświetlenie raportu klasyfikacji, który dostarcza bardziej szczegółowych informacji o wydajności modelu
print("Classification of the RF = \n",classification_rep_rf)
# Wyświetlenie macierzy pomyłek, która pomaga zrozumieć, jak model klasyfikuje różne klasy
print("Matrix of the RF = \n",conf_matrix_rf,"\n")



#6 Obliczanie waznosci

# Obliczanie ważności cech w modelu Random Forests
feature_validity = RF_model.feature_importances_
# Pobranie nazw cech z ramki danych X
feature_names = X.columns
# Tworzenie słownika przypisującego ważność cech do ich nazw
feature_importance_dict = dict(zip(feature_names, feature_validity))
# Sortowanie ważności cech malejąco
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Wyświetlenie nagłówka informującego o wydruku ważności cech
print("Ważność cech:")
# Iteracja przez posortowaną listę ważności cech
for feature, importance in sorted_feature_importance:
    # Wyświetlenie nazwy cechy i jej ważności
    print(f"{feature}: {importance}")

# Ustalenie rozmiaru wykresu
plt.figure(figsize=(10, 6))
# Generowanie wykresu słupkowego z ważnością cech
plt.barh(range(len(sorted_feature_importance)), [val[1] for val in sorted_feature_importance], align='center')
# Dodanie etykiet osi Y z nazwami cech
plt.yticks(range(len(sorted_feature_importance)), [val[0] for val in sorted_feature_importance])
# Ustalenie tytułu wykresu
plt.title('Ważność cech')
# Wyświetlenie wykresu ważności cech
plt.show()


#7
# Przykładowy model
RF_model = RandomForestClassifier()

# Liczba podziałów (kropli) w walidacji krzyżowej
n_splits = 5

# Utwórz obiekt do przeprowadzania walidacji krzyżowej
cv = KFold(n_splits=n_splits, shuffle=True, random_state=5)

# Wykonaj walidację krzyżową
scores = cross_val_score(RF_model, X, y, cv=cv, scoring='accuracy')  # Możesz użyć innych metryk oceny

# Wyświetl wyniki
print("Wyniki walidacji krzyżowej",scores)
print("Średnia dokładność",scores.mean())

#8
# Przestrzeń hiperparametrów do przeszukania
param_grid = {
    'C': np.logspace(-3, 3, 7),  # Parametr C w SVM
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Rodzaj jądra
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))  # Parametr gamma w jądrze rbf
}

# Tworzenie modelu SVM
SVM_model = SVC()

# Random Search
random_search = RandomizedSearchCV(SVM_model, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=-1, random_state=5, verbose=1)

# Trenowanie modelu z wykorzystaniem Random Search
random_search.fit(X_train, y_train)

# Najlepszy znaleziony zestaw hiperparametrów
best_params = random_search.best_params_

# Najlepszy model
best_model = random_search.best_estimator_

# Ocena najlepszego modelu na danych testowych
y_pred = best_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

# Wyświetlenie wyników
print("\n Najlepsze hiperparametry:", best_params)
print("Dokładność najlepszego modelu:", accuracy)