# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df=pd.read_csv(path)

df.head()

print(df.info)
print(df["INCOME"].head())
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
df[cols] = df[cols].replace({'\$': '', ',': ''}, regex=True)
X=df.drop(columns=["CLAIM_FLAG"],axis=1)
y=df["CLAIM_FLAG"]

count=y.value_counts()

print("value_counts of target(y):",count)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3, random_state = 6)



# Code ends here


# --------------
# Code starts here
cols=['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

X_train[cols]=X_train[cols].astype(float)
X_test[cols]=X_test[cols].astype(float)

print(X_train.isnull().sum())
print(X_test.isnull().sum())


# Code ends here


# --------------
# Code starts here
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)

y_train=y_train[X_train.index]
y_test=y_test[X_test.index]

cols=["AGE","CAR_AGE","INCOME","HOME_VAL"]
X_train[cols].fillna(X_train[cols].mean,inplace=True)
X_test[cols].fillna(X_train[cols].mean,inplace=True)

print(X_test.isnull().sum())
print(X_train.isnull().sum())

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for x in columns:
    le=LabelEncoder()
    X_train[x]=le.fit_transform(X_train[x].astype(str))
    X_test[x]=le.transform(X_test[x].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model=LogisticRegression(random_state=6)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

score=accuracy_score(y_test,y_pred)

print("Accuracy score:%.2f"%score)


# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state=9)
X_train,y_train=smote.fit_sample(X_train,y_train)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)




# Code ends here


# --------------
# Code Starts here
model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

score=accuracy_score(y_test,y_pred)
print("accuracy_score:%.2f"%score)

# Code ends here


