import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import potrzebnych bibliotek

# wczytanie rekordów z bazy .csv
data = pd.read_csv(r'diabetes.csv')

# wyswietlenie informacji o bazie
data.info()

# usuniecie wierszy z brakujacymi danymi (null)
data.dropna()

# wyswietlenie pierwszych wierszy danych
data.head()

# szybki przeglad danych
data.describe()

# data["Pregnancies"][0]
# data.iloc[0]
# data.columns

# nowy wykres o wymiarach 10x6 cali
plt.figure(figsize=(10, 6))

# histogram wieku pacjentow, rozklad wieku , parametr kde: dodanie wygładzonej krzywej pokazujaca ogolny ksztalt
sns.histplot(data["Age"], kde=True)

# etykieta osi x jako wiek
plt.xlabel('Age')

# etykieta osi y jako liczba pacjentow
plt.ylabel("Number of patients")

# wyswietlenie wykresu
plt.show()


# wykres o wymiarach 12 x 8 cali
plt.figure(figsize=(12, 8))

#wykres heatmapy parametr cmap: kolor wykresu, fmt: okreslenie dokladnosci do 2 miejsc po przecinku
sns.heatmap(data.corr(), annot=True, cmap="jet", fmt='.2f')

# wyswietlenie wykresu
plt.show()

# Przypisanie do zmiennej X ramkę danych i usuwa tym samym kolumne outcome z bazy
X = data.drop('Outcome', axis=1)
# Przypisanie do zmienney Y kolumny outcome czyli to co chcemy przewidywac
Y = data["Outcome"]
X.head()

# normalizacja

# stworzenie instancji obiektu StandardScaler
scaler = StandardScaler()

# przypisanie wyniku obliczen średniej i odchylenia standardowego dla cech X do zmiennej X_1
X_1 = scaler.fit_transform(X)

# przypisanie ramki danych z znormalizowanymi danymi oprocz ostatniej kolumny [:-1]
X_norm = pd.DataFrame(X_1, columns=data.columns[:-1])

# podstawowe informacje
X_norm.describe()

# podzial danych na zbiory treningowe i testowe, parametr test_size: proporcja miedzy zbiorem testowym a treningowym
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# stworzenie instancji obiektu sluzacego do klasyfikacji danych, parametr ozancza liczbe najblizszych sasiadow
knn_lib = KNeighborsClassifier(3)

# uczenie, dopasowanie modelu KNN do danych trenigowych
knn_lib.fit(X_train, y_train)
# ocena dokladnosci modelu na danych testowych
accuracy_lib = knn_lib.score(X_test, y_test)


# funkcja sluzaca do okreslania dokladnosci klasyfikacji
def calculate_accuracy(y_true, y_pred):
    # sumuwanie tylko wtedy gdy zwracana wartosc true
    corrected = sum(y_true == y_pred)
    # zwracamy stosunek poprawnych klasyfikacji przez calkowita liczbe
    return corrected / len(y_true)


# wlasna klasa KNN
class KNN:
    # konstruktor przyjmujac argument k: liczba najblizszych sasiadow
    def __init__(self, k):
        self.k = k

    # metoda sluzaca do przypisania modelu do danych treningowych
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # metoda sluzaca do przewidywania klasyfikacji
    def predict(self, X_train):
        # utworzenie pustej listy
        predictions = []

        # iteracja po kazdym wierszu
        for x in X_test.values:
            # odlegosc miedzy iterowanym wierszem a wszystkimi punktami w x_train
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train.values]
            # wybiera k najblizszych sasiadow po wczesniejszym posortowaniu
            k_indices = np.argsort(distances)[:self.k]
            # lista etykiet najblizszych sasiadow
            k_labels = [self.y_train.iloc[i] for i in k_indices]
            # wybranie najczesciej wystepujacej klasy
            most_common = max(set(k_labels), key=k_labels.count)
            # dodanie przewidywanej klasy do listy
            predictions.append(most_common)

        # zwracamy liste
        return predictions


# stworzenie instancji naszej wlasnej klasy KNN, z liczba najblizszych sasiadow=3
knn_withoutlib = KNN(3)

# doposowanie modelu do danych trenigowych
knn_withoutlib.fit(X_train, y_train)

# przewidywanie klas dla danych testowych
y_pred = knn_withoutlib.predict(X_test)

# obliczenie dokładnosci za pomoca naszej funkcji
accuracy_withoutlib = calculate_accuracy(y_test, y_pred)

if accuracy_withoutlib < accuracy_lib:
    print('Lepsza dokladnosc klasyfikatora zewnetrznej biblioteki')
elif accuracy_withoutlib > accuracy_lib:
    print('Lepsza dokladnosc klasyfikatora naszej implementacji')
else:
    print('Zaciety remis:)')


# dodać komentarze do kodu
# X_norm włozyc do podziału i zobaczyc jak sie uczy na zbiorze znormalizowanym
# dodać komentarrz jaki klasyfikator jest lepszy
# utworzyc tabele zbioru miekkiego
# zastosowac inferencje/wnioskowanie miekkie

# realizacja zadan w oddzielnym pliku excelu:)
