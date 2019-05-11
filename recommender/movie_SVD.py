from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.33)

algo = SVD()

algo.fit(trainset)
predictions = algo.test(testset)

accuracy.rmse(predictions)
