import matplotlib.pyplot as plt
from stocker import Stocker

amazon = Stocker('AMZN')

amazon.plot_stock()
# amazon.plot_stock(stats=['Daily Change'])

model, model_data = amazon.create_prophet_model(days=90)
model.plot_components(model_data)
plt.show()
amazon.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])

amazon.changepoint_prior_scale = 0.05
amazon.weekly_seasonality = True

amazon.evaluate_prediction()
