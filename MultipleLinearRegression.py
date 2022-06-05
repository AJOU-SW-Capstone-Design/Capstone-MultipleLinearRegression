from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

train=pd.read_csv('TrainData.csv')
test=pd.read_csv('TestData.csv')
X_train=train[['X1','X2','X3']]
y_train=train['Y']
X_test=test[['X1','X2','X3']]
y_test=test['Y']

model = LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
#R square
print("정확도",model.score(X_test, y_test))

y_predict = model.predict(X_test)
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual Total Time")
plt.ylabel("Predicted Total Time")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()