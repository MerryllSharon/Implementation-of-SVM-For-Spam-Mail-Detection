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

```
## Output:

### data_head():
![Code_vbNDleg2wo](https://github.com/user-attachments/assets/77fe6a77-1b60-498a-b24a-d36bf96485bb)


### data.isnull().sum():
![Code_rGpoFZYxYv](https://github.com/user-attachments/assets/30d184b9-01a4-4e85-9e05-d25aef221213)


### accuracy:
![Code_PzHfASbsbY](https://github.com/user-attachments/assets/427ccdc5-055f-4a69-8aaa-93cb5f05b005)

### Y_Pred:
![Code_7tVrrnX8TZ](https://github.com/user-attachments/assets/83d1797d-6874-4e0e-9082-4e538132d019)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
