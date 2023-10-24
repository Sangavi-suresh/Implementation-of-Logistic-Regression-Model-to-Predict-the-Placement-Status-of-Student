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
```

## Output:
# HEAD OF THE DATA :
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/cd2a61ae-4614-446a-bf05-c2a12ad5d089)


# COPY HEAD OF THE DATA :
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/422ccdd0-4bf9-4fe7-95c6-be60d8a0efe5)

# NULL AND SUM:
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/fc128242-5ca8-44ec-bcd8-1b22d9cb3292)

# DUPLICATED:
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/e23e5e3b-1a6f-4727-9b3e-963bd1de5c44)

# X VALUE:
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/e4ee3ec5-f749-42f8-834a-8d7bf6ad76d4)

# Y VALUE:
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/a5819f94-ddb1-4912-9108-d46fa68d3be0)

# PREDICTED VALUES:
![image](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/ec6a48d4-926e-481e-b9f8-849c1efb038d)

# ACCURACY:
![accura](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/19fa1d3b-d5f3-45e9-89a0-9136887ed99e)


# CONFUSION MATRIX :

![confusion](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/20e9a3ef-2c4c-4268-9933-84fdc5d131b1)

# CLASSIFICATION REPORT :
![clasrepo](https://github.com/Sangavi-suresh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541861/5d61e3b1-0efd-4e1a-8c21-0a5f954d182d)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
