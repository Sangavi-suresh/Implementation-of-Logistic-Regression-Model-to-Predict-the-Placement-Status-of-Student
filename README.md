# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1 :
Import the standard libraries such as pandas module to read the corresponding csv file.

Step 2 :
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

Step 3 :
Import LabelEncoder and encode the corresponding dataset values.

Step 4 :
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

Step 5 :
Predict the values of array using the variable y_pred.

Step 6 :
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

Step 7 :
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

Step 8:
End the program.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:sangavi suresh 
RegisterNumber:  212222230130
*/
```
import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


## Output:
# PLACEMENT DATA :
![headdata](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/054e02e0-b394-4939-b3e9-6ecef66fe32a)

# SALARY DATA :
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/aa56619b-3ef9-469f-a21e-1d8a42edfcd4)

# CHECKING THE NULL FUNCTION:
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/09cd86aa-ad67-4552-ba4f-0ffb2536cd2b)

# Data Duplicate:
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/e2a5d1d3-4704-4b42-b613-424ab0c9b604)

#PRINT DATA :
![xvalue](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/9935f980-326f-4545-be49-306edbcd5a73)


# DATA STATUS :
![yvalue](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/3b13ccd1-9ec6-4180-b40f-4e7dba7dc6d3)


# Y_PREDICTED ARRAY:
![predvaluse](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/8481c836-8833-4818-a762-612dc17189cd)


# ACCURACY VALUES :
![accura](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/19fa1d3b-d5f3-45e9-89a0-9136887ed99e)


# CONFUSION MATRIX :

![confusion](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/20e9a3ef-2c4c-4268-9933-84fdc5d131b1)

# CLASSIFICATION REPORT :
![clasrepo](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/5d61e3b1-0efd-4e1a-8c21-0a5f954d182d)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
