''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split


#how many sameples and How many features?
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

##diabetes = print("Let's dia-beat this")

print(diabetes.data.shape)


# What does feature s6 represent?
# glu, blood sugar level
#print(diabetes.DESCR)




#print out the coefficient
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)

linear_regression = LinearRegression() 

linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)

#print out the intercept

print(linear_regression.intercept_)

# create a scatterplot with regression line

import seaborn as sns

axes = sns.scatterplot(
    data=diabetes,
    x="age",
    y="Blood Sugar Level",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10,70) # Scale y-axis

import numpy as np

x = np.array([min(diabetes.data.values), max(diabetes.data.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)

plt.show()

