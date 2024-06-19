import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

weather_data = pd.read_csv('seattle-weather.csv')

X = weather_data[['precipitation', 'temp_max', 'temp_min', 'wind']]
Y = weather_data['weather']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

pickle.dump(model, open('ml_model.pkl', 'wb'))