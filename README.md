# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
developed by madhanKumar j
reg no 2305001016
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('/content/ex1.csv')
df.head()

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')

x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,Y_train)

X_train
Y_train

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')

m=lr.coef_
m

b=lr.intercept_
b

pred=lr.predict(X_test)
pred

X_test

Y_test

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,pred)
print(f"Mean Squared Error (MSE): {mse}")
```

## Output:
![image](https://github.com/user-attachments/assets/effb36fe-503a-4871-88bb-ab1a60c0848e)

![image](https://github.com/user-attachments/assets/0e3f65a6-1e02-4236-8399-1556b4260110)

![image](https://github.com/user-attachments/assets/e927184c-9585-4812-99b8-7db332cf2e1a)

![image](https://github.com/user-attachments/assets/deddffae-cb84-47e5-bc35-b6a107c1eeea)

![image](https://github.com/user-attachments/assets/9355c481-353f-46e8-a226-085b56c976d7)

![image](https://github.com/user-attachments/assets/7a3e148f-43f5-428b-ae5a-b5c1128adb05)

![image](https://github.com/user-attachments/assets/b5fc3005-81cf-4565-9f05-3a5dbe0c68c5)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
