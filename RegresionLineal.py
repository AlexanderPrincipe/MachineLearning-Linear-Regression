
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split 

data= np.genfromtxt("housing.txt",dtype='O',delimiter="\t",skip_header=True)
#convert from categorical string to float
X=data[:,0].astype(np.float)
y=data[:,1].astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, y)
lr=linear_model.LinearRegression()
lr.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
print "intercept ",lr.intercept_
yfit=lr.predict(X_test.reshape(-1,1))
plt.scatter(X_test,y_test)
plt.plot(X_test,yfit)
