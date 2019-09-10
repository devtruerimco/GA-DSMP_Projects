# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here
df=pd.read_csv(path)

df.head()

X=df.drop(columns=["attr1089"],axis=1)
y=df["attr1089"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)

#Scaling
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#Initiate a Logistic Regression model
lr=LogisticRegression()

#Fitting model on train data
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

#ROC AUC score of prediction
roc_score=roc_auc_score(y_test,y_pred)
print("roc_score:%.2f"%roc_score)


# --------------
from sklearn.tree import DecisionTreeClassifier

#Initiate DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=4)

#Fitting model on train data
dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)

#ROC AUC score of prediction
roc_score=roc_auc_score(y_test,y_pred)
print("roc_score:%.2f"%roc_score)



# --------------
from sklearn.ensemble import RandomForestClassifier


# Code strats here

#Initiate a Random Forest
rfc=RandomForestClassifier(random_state=4)

#Fitting model on train data
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

#ROC AUC score of prediction
roc_score=roc_auc_score(y_test,y_pred)
print("roc_score:%.2f"%roc_score)



# Code ends here


# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier


# Code starts here

#Initiate BaggingClassifier
bagging_clf=BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,max_samples=100,random_state=0)

#Fitting model on train data
bagging_clf.fit(X_train,y_train)


#accuracy score of test data
score_bagging=bagging_clf.score(X_test,y_test)
print("score_bagging:%.2f"%score_bagging)



# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here
voting_clf_hard=VotingClassifier(estimators=model_list,voting="hard")

#Fitting model on train data
voting_clf_hard.fit(X_train,y_train)


#accuracy score of test data
hard_voting_score=voting_clf_hard.score(X_test,y_test)
print("hard_voting_score:%.2f"%hard_voting_score)



# Code ends here


