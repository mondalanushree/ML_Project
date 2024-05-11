import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
df=sns.load_dataset('mpg')
df.isnull().sum()
df.dropna(inplace=True)
x=df[['displacement'	,'horsepower',	'weight'	,'acceleration']]
y =df.mpg
X_train, X_test, Y_train, Y_test =train_test_split(x, y, test_size =0.15, random_state=42)
#from ctypes import LibraryLoader


model = LinearRegression()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

model2= DecisionTreeRegressor(criterion="poisson",random_state=0)
model2.fit(X_train,Y_train)
model2.score(X_test,Y_test)
filename = 'mpg_regression.sav'
pickle.dump(model, open(filename, 'wb'))