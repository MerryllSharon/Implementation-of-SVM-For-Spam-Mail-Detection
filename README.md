# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.


## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Merryll S Sharon
RegisterNumber:  212222080033
*/
```
```python
import pandas as pd
data= pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test= cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train , y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy= metrics.accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix, classification_report
con= confusion_matrix(y_test, y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:

### data_head():


### data.isnull().sum():


### accuracy:


### Confusion matrix:

### Classification report:


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
