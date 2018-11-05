import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

n_users, n_items = data['train'].shape


def sample_recommendation(model, data, user_id):
    known_positives = data['item_labels'][data['train'].tocsr()[
        user_id].indices]
    scores = model.predict(user_id, np.arange(n_items))
    top_items = data['item_labels'][np.argsort(-scores)]
    print("Liked Already:")
    for x in known_positives[:3]:
        print(x)
    print("Recommended:")
    for x in top_items[:3]:
        print(x)


print(sample_recommendation(model, data, 17))
