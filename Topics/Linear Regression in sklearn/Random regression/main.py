import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("./data/dataset/input.txt")
X_train = df.iloc[:-70, :4]
X_test = df.iloc[-70:, :4]
y_train = df.target[:-70]
y_test = df.target[-70:]
RM = LinearRegression()
RM.fit(X_train, y_train)
print(round(RM.intercept_,3))




