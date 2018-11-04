from tpot import TPOTClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
             'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'Class']
df = pd.read_csv('gamma.data', names=col_names)
shuffle = df.iloc[np.random.permutation(len(df))]
df = shuffle.reset_index(drop=True)

class_map = {'g': 0, 'h': 1}

df['Class'] = df['Class'].map(class_map)
classes = df['Class'].values

training_indices, validation_indices = training_indices, testing_indices = train_test_split(df.index,
                                                                                            stratify=classes, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, verbosity=2)
tpot.fit(df.drop('Class', axis=1).loc[training_indices].values,
         df.loc[training_indices, 'Class'].values)
tpot.score(df.drop('Class', axis=1).loc[validation_indices].values,
           df.loc[validation_indices, 'Class'].values)
tpot.export('pipeline.py')
