from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd

iris = datasets.load_iris()
data = pd.DataFrame({
    'sepalLength': iris.data[:, 0],
    'sepalWidth': iris.data[:, 1],
    'petalLength': iris.data[:, 2],
    'petalWidth': iris.data[:, 3],
    'species': iris.target
})

X = data[['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']]
Y = data['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=150)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
