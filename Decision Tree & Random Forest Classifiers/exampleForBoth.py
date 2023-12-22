import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

df = pd.read_csv("C:\\Users\\deela\\Downloads\\creditcard.csv")
print(df)

print(df.info())

# Feature Importance/Selection
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

'''model = ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_)'''

cols = ['V17', 'V14', 'V12', 'V10', 'V11', 'V16', 'V18', 'V9', 'V4', 'V3', 'V7',
       'V21', 'V1', 'V26', 'Time', 'V2', 'V19', 'V8']
X_new = X[cols]

# Spliting the data into sets
skf = StratifiedKFold(n_splits=10)

for train_index, test_index in skf.split(X,y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

for train_index, test_index in skf.split(X_new,y):
  X_new_train, X_new_test = X_new.iloc[train_index], X_new.iloc[test_index]
  y_new_train, y_new_test = y.iloc[train_index], y.iloc[test_index]

# Model Selection

decision = DecisionTreeClassifier()
randomf = RandomForestClassifier()

# Hyper Parameter Tuning for RandomForestClassifier
n_estimators = [int(i) for i in np.linspace(100,1200,12)]
max_features = ['auto', 'sqrt']
max_depth = [int(i) for i in np.linspace(5,30,5)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]

parameters = {
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf
}

rf_model = RandomizedSearchCV(estimator=randomf,
                              param_distributions=parameters,
                              scoring='neg_mean_squared_error',
                              n_jobs=1,
                              cv=5,
                              verbose=2,
                              random_state=42
                              )

rf_model.fit(X_train,y_train)

#for random forest
randomf.fit(X_train,y_train)

y_pred = randomf.predict(X_test)

print(accuracy_score(y_test,y_pred))