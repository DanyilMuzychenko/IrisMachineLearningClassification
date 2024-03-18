# Classify-iris-flowers-using-machine-learning
Klasyfikacja kwiatów tęczówki przy użyciu różnych algorytmów uczenia maszynowego.


<p align="center">
      <img src="https://i.ibb.co/NSS0bc2/iris-2.webp" alt="Project Logo" width="746">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Engine-PyCharm%2023-B7F352" alt="Engine">
</p>

## About


Projekt zakłada wykorzystanie zestawu danych Iris, który jest dostępny publicznie i zawiera 150 rekordów. Celem jest stworzenie klasyfikatora opartego na uczeniu maszynowym, który będzie klasyfikował irysy na podstawie ich cech, takich jak długość i szerokość działki oraz długość i szerokość płatka.
</br>
W projekcie zastosowano takie modeli uczenia maszynowego:
- - `K-Nearest Neighbors (KNN)`
- - `Support Vector Machine (SVM)`
- - `Random Forest`
## Documentation

### Libraries
- `NumPy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### Przygotowanie danych
- Dane są wczytywane z pliku "iris.csv".
- Generowana jest macierz cech X i wektor etykiet y.
### Wizualizacja danych
- Tworzone są wykresy parami w celu wizualnej analizy zależności między cechami i rozkładami klas.
### K-Nearest Neighbours (KNN):
- Model KNN jest inicjowany liczbą sąsiadów (k=2).
- Model jest trenowany na danych treningowych.
- Przewidywania są dokonywane na danych testowych.
- Wyniki obejmują dokładność, raport klasyfikacji i macierz pomyłek.
### Maszyna wektorów nośnych (SVM):
- Model SVM z jądrem liniowym i parametrem C=2 jest inicjowany i trenowany.
- Przewidywania są dokonywane na danych testowych.
- Wyniki obejmują dokładność, raport klasyfikacji i macierz pomyłek.
### Random Forest:
- Model Random Forest ze 100 drzewami jest inicjowany i trenowany.
- Przewidywania są dokonywane na danych testowych.
- Obliczana jest ważność cech wykres ważności.
- Wyniki obejmują dokładność, raport klasyfikacji i macierz pomyłek.
### Analiza ważności atrybutów:
- Wyświetlana jest ważność każdej cechy i wykreślany jest wykres ważności.
### Walidacja krzyżowa i wyszukiwanie siatki:
- KFold służy do przeprowadzania walidacji krzyżowej.
- Wyszukiwanie siatki jest wykonywane dla modelu Random Forest w celu znalezienia optymalnych parametrów.
- Wyniki walidacji krzyżowej i optymalne parametry modelu są wyprowadzane.
### Przewidywanie na nowych danych:
- Prognozy są wykonywane dla trzech nowych zestawów danych.
## Developers

- Danyil Muzychenko (https://github.com/TheHallRide)
