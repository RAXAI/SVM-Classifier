import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#Load data
train_data = pd.read_csv('svmtrain.txt',sep=' ')
test_data  = pd.read_csv('svmtest.txt', sep=' ')
# split data in to feature and target arrays
X = np.array(train_data.drop('C',1))

y = np.array(train_data.drop(['X1','X2'],1)).reshape(-1,1).ravel()




#Converts test data DataFrame into numpy array
X_test= np.array(test_data.drop('C',1))




svm = SVC()


#paramter grid
parameters = {'kernel':['rbf'],'C':[0.0001,0.001,0.005,0.1,0.5,1, 10,100],'gamma':[0.0000001,0.000001,0.00001, 0.0001, 0.001,0.005 ,0.01, 0.1,1]}

#fitting the model
model = GridSearchCV(svm, parameters, cv=10)
model.fit(X,y)

#classes predicted from the model,using test data assingned to y_pred variable
y_pred = model.predict(np.array(test_data.drop('C',1)))

#calculates the average mean and std arcross all paramter combinations
mean = sum(model.cv_results_['mean_test_score'])/len(model.cv_results_['mean_test_score'])
std = sum(model.cv_results_['std_test_score'])/len(model.cv_results_['std_test_score'])
# Report the best parameters
print("Best CV params", model.best_params_)
print("Best CV accuracy", model.best_score_)
print('Average mean across all parameter combination: ',mean,'Average std across all parameter combinations: ',std)

#printing predicted values from test data

print('predcited classes',y_pred)
