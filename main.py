# predict a diagnosis by sonu


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('breast-cancer.csv')

df.head(1)

df.columns


df.drop(['id','Unnamed: 32'],axis=1,inplace=True)

df

df.describe()

df['diagnosis'].value_counts()

# 0 is a B
#1 is a M
diag=pd.get_dummies(df['diagnosis'],drop_first=True)

df.drop('diagnosis',inplace= True,axis=1)

df

df=pd.concat([df,diag],axis=1)

df

df.isnull().sum()

df.columns

x=df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]

y=df['M']

y.value_counts()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

lr.intercept_

prediction=lr.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,prediction)

from sklearn.metrics import classification_report

print(classification_report(y_test,prediction))
