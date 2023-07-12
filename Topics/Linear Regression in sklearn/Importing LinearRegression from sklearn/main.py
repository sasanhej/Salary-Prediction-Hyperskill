from sklearn.linear_model import LinearRegression
import numpy as np

RM = LinearRegression()
X = np.array([4, 6, 8, 10, 12, 14, 16]).reshape(7, 1)
y = np.array([2, 2, 3, 5, 5, 6, 6]).reshape(-1, 1)
print(X)
RM.fit(X, y)
print(RM.predict([[23]]))

