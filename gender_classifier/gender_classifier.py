from sklearn import tree
import pandas as pd

color = {'Neutral': 0, 'Warm': 1, 'Cool': 2}
music = {'Electronic': 0, 'Folk/Traditional': 1, 'Hip hop': 2,
         'Jazz/Blues': 3, 'Pop': 4, 'Rock': 5, 'R&B and soul': 6}
beverage = {'Doesn\'t drink': 0, 'Beer': 1,
            'Other': 2, 'Vodka': 3, 'Whiskey': 4, 'Wine': 5}
soft_drink = {'7UP/Sprite': 0, 'Coca Cola/Pepsi': 1, 'Fanta': 2, 'Other': 3}

# Import and Format Data
df = pd.read_csv('data.csv')
df['Favorite Color'] = df['Favorite Color'].map(color)
df['Favorite Music Genre'] = df['Favorite Music Genre'].map(music)
df['Favorite Beverage'] = df['Favorite Beverage'].map(beverage)
df['Favorite Soft Drink'] = df['Favorite Soft Drink'].map(soft_drink)

# X => Features
X = df[['Favorite Color', 'Favorite Music Genre', 'Favorite Beverage',
        'Favorite Soft Drink']]
# Y => Gender
Y = df['Gender']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# Predict
predict = clf.predict([[2, 4, 4, 1]])
print(predict)
