# Polynomial-regression

## Beginner level polynomial regression (Machine learning) 
---

import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]

y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)

plt.show()
---

![image](https://user-images.githubusercontent.com/100121721/166674536-6fc9e143-88da-4aad-9f14-4a6bf04e93d5.png)

---
from scipy import stats

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):

  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)

plt.plot(x, mymodel)

plt.show()
---

![image](https://user-images.githubusercontent.com/100121721/166674650-07098714-977e-4cd7-92a9-6dd7b1e8bf26.png)


Step by step Coding explanation:
1.	First import the libraries which are needed (matplotlib, SciPy)
2.	Creating both arrays to show the X and Y-axis.
3.	Executing the method such as data considered above
4.	A function is being created using the slope and intercept to return a value, run each value of the x array through the function that will create a new array with new values at the y-axis
5.	Draw the scatter plot and linear regression as well and display the pictures.


Python:
In the example below, the x-axis represents the hours of the day and the y-axis represents speed. We have registered the age and speed of 18 trains as they were passing a railway station. Let us see if the data we collected could be used in polynomial regression (train speed and the time of the day passing occurrence periods are being collected)


import numpy

import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]

y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)

plt.plot(myline, mymodel(myline))

plt.show()

![image](https://user-images.githubusercontent.com/100121721/166674907-8bed02e1-eab5-418a-bd53-a1c982fbcd37.png)

---

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values
---
![image](https://user-images.githubusercontent.com/100121721/166667677-d2f8a3f5-b598-4705-8559-abdbbc39a3df.png)

---
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')

plt.plot(X, lin_reg.predict(X), color = 'blue')

plt.title('Truth or Bluff (Linear Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
---
![image](https://user-images.githubusercontent.com/100121721/166667820-8bf30350-5154-43e5-9e35-13d5ca45f362.png)

---
plt.scatter(X, y, color = 'red')

plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('Truth or Bluff (Polynomial Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
---

![image](https://user-images.githubusercontent.com/100121721/166667875-52d4496a-acbb-4c19-8e04-8750cced18da.png)

 predictions

---
lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

---

![image](https://user-images.githubusercontent.com/100121721/166675092-cacc2042-9cfd-4786-b012-f9ac1c2d37c1.png)

