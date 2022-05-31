from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("Umandong100.csv")

X = data[['x1','x2','x3']]
y = data[['y']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=10000)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

#R square
print("정확도",mlr.score(X_test, y_test))
print(mlr.predict([[6,18,3]]))

y_predict = mlr.predict(X_test)
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual Total Time")
plt.ylabel("Predicted Total Time")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()