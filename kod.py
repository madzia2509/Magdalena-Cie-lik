import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

data = pd.read_csv("Mroz.csv", header=0)
print(data)
print(data.columns)

print(data.isnull().sum())

print(type(data))

print(data.info())

print(data.corr())

plt.figure(figsize=(17, 17))
sns.heatmap(data.corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1)
plt.title("Macierz korelacji")
plt.show()

print(data.describe().T)

pd.DataFrame(data['hoursw']).hist()
plt.show()

print(data.city.value_counts())

city_no = data[data.city == 'no']
city_yes = data[data.city == 'yes']

print(city_no.describe())
print(city_yes.describe())

city_yes = city_yes.iloc[:,:-1].copy()
print(city_yes)

# Zmiana 'yes','no' na 1, 0
data.work = pd.Series(np.where(data.work.values == 'yes', 1, 0), data.index)
data.city = pd.Series(np.where(data.city.values == 'yes', 1, 0), data.index)

scaler = StandardScaler()
demo_scaled = scaler.fit_transform(data)

hier_clust = linkage(demo_scaled, method='ward')
plt.figure(figsize=(8, 6))
plt.title('Wykres grupowania', fontsize=20)
plt.ylabel('Odległość', fontsize=13)
plt.xlabel('Obserwacje', fontsize=13)
dendrogram(hier_clust, show_leaf_counts=False, truncate_mode='level', p=5, no_labels=True)
plt.show()

wcss = {}
for i in range(10, int(data.shape[0]/10), 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(demo_scaled)
    wcss[i] = kmeans.inertia_

plt.figure(figsize=(8, 5))
plt.plot(list(wcss.keys()), list(wcss.values()), marker='o', linestyle='--')

plt.xlabel('Numer klastra', fontsize=13)
plt.ylabel('WCSS', fontsize=13)
plt.title('Metoda k-średnich', fontsize=15)

plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data_class_tree = data.copy()

y = data_class_tree.pop('work')
X = data_class_tree

clf = DecisionTreeClassifier(max_depth=4)

# Podział danych na dane testowe oraz treningowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Uczenie
clf = clf.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import plot_tree

plot_tree(clf, feature_names=X.columns, class_names=['work', 'doesn\'t work'])
plt.title('Decision tree classifier')
plt.show()

data_class_tree = data.copy()

y = data_class_tree.pop('city')
X = data_class_tree

# clf = DecisionTreeClassifier(max_depth=4, splitter='random', criterion='log_loss')
clf = DecisionTreeClassifier(max_depth=4, splitter='best', criterion='gini', random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

plot_tree(clf, feature_names=X.columns, class_names=['city', 'not city'])
plt.title('Decision tree classifier')
plt.show()

from copy import deepcopy

data_class_tree = data.copy()

y = data_class_tree.pop('city')
X = data_class_tree

clf = DecisionTreeClassifier(max_depth=4, splitter='best', criterion='gini')

accuracy = 0
best_clf = None

for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    current_accuracy = metrics.accuracy_score(y_test, y_pred)
    if current_accuracy > accuracy:
        accuracy = current_accuracy
        best_clf = deepcopy(clf)

print(accuracy)

plot_tree(best_clf, feature_names=X.columns, class_names=['city', 'not city'])
plt.title('Decision tree classifier')
plt.show()


from dtreeviz import model as dtreemodel

viz = dtreemodel(best_clf, X, y,
                 target_name="city",
                 feature_names=X.columns)

viz.view().show()
from sklearn.tree import DecisionTreeRegressor

data_regression_tree = data.copy()

y = data_regression_tree.pop('hearnw')
X = data_regression_tree

regressor = DecisionTreeRegressor(max_depth=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

print('Zbiór testowy\n', X_test)
print('Predykcje:', predictions)

plot_tree(regressor, feature_names=X.columns)
plt.title('Decision tree regressor')
plt.show()


viz = dtreemodel(regressor, X, y,
                 target_name="hearnw",
                 feature_names=X.columns)

viz.view().show()
