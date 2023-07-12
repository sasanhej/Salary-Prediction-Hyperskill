import os
import requests
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')


def regmodel(X, y, test_size=0.3, random_state=100, drop=list([])):
    regdict = {}
    if drop != list([]):
        X = X.drop(columns=drop)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    regdict['intercept'] = model.intercept_
    regdict['coef'] = model.coef_
    regdict['X_train'] = X_train
    regdict['X_test'] = X_test
    regdict['y_train'] = y_train
    regdict['y_test'] = y_test
    regdict['y_pred'] = y_pred
    regdict['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)
    return regdict


def plotreg(X, y, dic, i=1):
    Xp = X**i
    yp = y
    plt.scatter(Xp, yp)
    plt.title("Rating-Salary power of: {}".format(i))
    plt.xlabel("Rating")
    plt.ylabel("Salary")
    a = dic[i]['coef']
    b = dic[i]['intercept']
    g = np.linspace(max(Xp.min()[0], Xp.iloc[yp.idxmin()][0]), min(Xp.max()[0], Xp.iloc[yp.idxmax()][0]), 1000)
    h = a*g+b
    plt.plot(g, h, '-r')
    plt.show()


'''
X = pd.DataFrame(data.rating)
y = pd.Series(data.salary)

Stage 1/5
print(f'{power[1]["intercept"]:.5f}', end=" ")
print(f'{power[1]["coef"][0]:.5f}', end=" ")
print(f'{power[1]["MAPE"]:.5f}')

Stage 2/5
power = {}
MAPE=list([])
for i in range(1, 5):
    power[i] = regmodel(X**i,y)
    MAPE.append(power[i]['MAPE'])
    plotreg(X, y, power, i)
print(f'{min(MAPE):.5f}')

#Stage 3/5
target = 'salary'
X = data.drop(columns=[target])
y = pd.Series(data[target])
print(*regmodel(X,y)['coef'], sep=", ")

plt.title("Salary predictions vs. Salary real data")
plt.xlabel("Predictions")
plt.ylabel("Real Data")
plt.scatter(regmodel(X,y)['y_pred'], regmodel(X,y)['y_test'])
g = np.linspace(max(regmodel(X,y)['y_pred'].min(),regmodel(X,y)['y_test'].min()), min(regmodel(X,y)['y_pred'].max(),regmodel(X,y)['y_test'].max()), 10)
plt.plot(g, g, '-r')
plt.show()

#Stage 4/5
high_corel = list([])
corel_matrix = data.drop(columns='salary').corr()
for i in itertools.combinations(corel_matrix.columns, 2):
    if abs(corel_matrix.loc[i]) > 0.2:
        high_corel.append([i[0], i[1]])
        if i[0] not in high_corel:
            high_corel.append(i[0])
        if i[1] not in high_corel:
            high_corel.append(i[1])
target = 'salary'
MAPE=[]
X = data.drop(columns=[target])
y = pd.Series(data[target])
for i in high_corel:
    iter = regmodel(X, y, drop=i)
    MAPE.append(iter['MAPE'])
print(f'{min(MAPE):.5f}')
'''
#Stage 4/5
target = 'salary'
drop = ['age', 'experience']
X = data.drop(columns=[target])
y = pd.Series(data[target])

middle = regmodel(X, y, drop=drop)

y_pred_0 = np.array([0 if i < 0 else i for i in middle['y_pred']])
y_pred_median = np.array([middle['y_train'].median() if i < 0 else i for i in middle['y_pred']])

MAPE_0 = mean_absolute_percentage_error(middle['y_test'], y_pred_0)
MAPE_median = mean_absolute_percentage_error(middle['y_test'], y_pred_median)
print(f'{min(MAPE_0,MAPE_median):.5f}')
