# 50_startups
##Regression_Problem_With_All_Algorithm
# STEP 1 : Importing Libraries and Dataset

import os
os.chdir("C:\\Users\\dhanush.g\\Desktop\\Imarticus python\\13th March 2021-20210313T114554Z-001\\13th March 2021")
os.getcwd()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset = pd.read_csv('50_Startups.csv')
dataset.head()

# STEP 2 : Data Cleaning and Preprocessing 

# missing data
pd.DataFrame(dataset).isnull().any()

# missing data
pd.DataFrame(dataset).isnull().sum()

# check the number of features in the dataset
print(len(dataset))
print(len(dataset.columns))
# check the data type of each columns
print(dataset.dtypes)

# Find the information of the raw dataset

dataset.info()

# STEP 3 : Finding Correlation

# to find the relation with Dependent variable vs Independant variable
features = dataset.iloc[:,0:3].columns.tolist()
features

target = dataset.iloc[:,4].name
target

from scipy.stats import pearsonr 

# Finding correlation of Profit with other variables to see how many variables are
    # strongly correlated with Profit
    
correlations = {}
for i in features:
    data = dataset[[i, target]]
    x1 = data[i].values
    x2 = data[target].values
    key = i + "Vs" + target
    correlations[key] = pearsonr(x1,x2)[0]

correlations

data_correlations = pd.DataFrame(correlations, index = ['Value']).T
data_correlations.loc[data_correlations['Value'].abs().
                      sort_values(ascending=False).index]

plt.figure(figsize=(30,8))
sns.heatmap(dataset.corr(), cmap='coolwarm', annot=True)
plt.show()

# STEP 4 : EDA (EXploratory Data Analysis) or Data Visualisation

from scipy.stats import stats
from scipy.stats import norm, skew

sns.lmplot(x='R&D Spend', y='Profit', data=dataset)

plt.scatter(x='R&D Spend', y='Profit', data=dataset)

plt.figure(figsize = (16,8))
sns.boxplot(x='State', y='Profit', data = dataset)
plt.show()

plt.figure(figsize = (16,8))
sns.barplot(x='State', y='Profit', data = dataset)
plt.show()

sns.pairplot(dataset, hue='State')

sns.pairplot(dataset, hue='State', diag_kind='hist')

# it's little complex but hope you understand
sns.distplot(dataset['Profit'], fit=norm);

# fitted with some parameter by using mu and sigma

(mu, sigma) = norm.fit(dataset['Profit'])

plt.legend(['Normal Dist. ($\mu=$ {:.2f} and $\sigma=${:.2f})'.format(mu, sigma)],
          loc='best')
plt.ylabel('Frequency')
plt.title("Profit Distribution")



# STEP 5 : Split the data into Ind and Dv

x = dataset.iloc[:,0:4].values
dataset.head()
x

y = dataset.iloc[:,4].values
y

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('one_hot_encoder',OneHotEncoder(categories='auto',),[3])],  
                       remainder='passthrough')

# in default categories = auto
onehot_x= np.array(ct.fit_transform(x), dtype=np.str)
# or the code in next line.


# please fix this bug by using different package

# STEP 6 : Spliting the data into training and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=101)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#############################################
# STEP 7 : Applying Machine Learning Model 

# Please use Linear Regression Model

# Decision Tree
# Random Forest - Bagging
# Ensemble Techniques
    # GradientBoostingRegressor
    # AdaBoostRegressor
    # LightBM
    
# Support Vector Machine


# ML 1 : Decision Tree

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import explained_variance_score
from time import time
# Model Building
start = time()
decision = DecisionTreeRegressor()
decision.fit(x_train, y_train)
decc = decision.score(x_test,y_test)

# Prediction
decpredict = decision.predict(x_test)


# explained_variance_score - comparing pred vs actual
# confusion_matrix - comparing actual vs pred

# Score / Accuracy
exp_dec = explained_variance_score(decpredict, y_test)
end = time()
train_time_dec = end-start

exp_dec


plt.figure(figsize=(15,8))
plt.scatter(y_test,decpredict, c = 'blue')
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.show()

plt.figure(figsize=(17,8))
plt.plot(y_test, label = "Test")
plt.plot(decpredict, label = "predict")
plt.show()

# ML 2 : RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
start = time()
rand_regr = RandomForestRegressor(n_estimators = 400, random_state=0)
rand_regr.fit(x_train, y_train)
random = rand_regr.score(x_test, y_test)

train_test_rand = end - start
predict_rand = rand_regr.predict(x_test)
exp_rand = explained_variance_score(predict_rand , y_test)
end = time()
train_time_rand = end-start

exp_rand

plt.figure(figsize=(15,8))
plt.scatter(y_test,predict_rand, c = 'blue')
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.show()

plt.figure(figsize=(17,8))
plt.plot(y_test, label = "Test")
plt.plot(predict_rand, label = "predict")
plt.show()

# ML 3 : GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingRegressor
start = time()
est = GradientBoostingRegressor(n_estimators = 400, max_depth=5, loss='ls', min_samples_split=2, learning_rate=0.1).fit(x_train, y_train)
gradient = est.score(x_test, y_test)
# Loss function - MAE, MAPE, MSE, RME

end = time()
train_test_est = end - start
predict_est = est.predict(x_test)
exp_est = explained_variance_score(predict_est, y_test)


exp_est

plt.figure(figsize=(15,8))
plt.scatter(y_test,predict_est, c = 'blue')
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.show()

plt.figure(figsize=(17,8))
plt.plot(y_test, label = "Test")
plt.plot(predict_est, label = "predict")
plt.show()

# ML 4 : AdaBoostRegressor

from sklearn.ensemble import AdaBoostRegressor
start = time()
ada = AdaBoostRegressor(n_estimators=50, learning_rate=0.2, loss='exponential').fit(x_train, y_train)
adab = ada.score(x_test, y_test)
# Loss function - MAE, MAPE, MSE, RME

end = time()
train_test_ada = end - start
predict_ada = ada.predict(x_test)
exp_ada = explained_variance_score(predict_ada, y_test)

exp_ada 

# ML 5 : Support Vector Machine 

from sklearn.svm import SVR
start = time()
svr = SVR(kernel='linear')
svr.fit(x_train, y_train)
end = time()
train_time_svr = end-start
svr1 = svr.score(x_test, y_test)
prediction_svr = svr.predict(x_test)
exp_svr = explained_variance_score(prediction_svr, y_test)


exp_svr

# ML 6 : LinearRegression

from sklearn.linear_model import LinearRegression
start = time()
regressor = LinearRegression()
regressor.fit(x_train, y_train)
end = time()
train_time_linear = end-start
regressor1 = regressor.score(x_test, y_test)
prediction_linear = regressor.predict(x_test)
exp_linear = explained_variance_score(prediction_linear, y_test)

exp_linear

# STEP 8 : Model Comparision

### Model Comparision on the basis of Model's Accuracy Score and Explained Variance score of different models

model_validation = pd.DataFrame({
    'Model':['Decision Tree','Random Forest','Gradiant Boosting','AdaBoost',
            'Support Vector Machine','Linear Regression'],
    'Score': [decc,random,gradient,adab,svr1,regressor1],
    'Variance Score': [exp_dec,exp_rand,exp_est,exp_ada,exp_svr,exp_linear]
    
    
})


model_validation.sort_values(by='Score', ascending=False)

# STEP 9 : Analysing training time for each model has taken

Model = ['Decision Tree','Random Forest','Gradiant Boosting','AdaBoost',
            'Support Vector Machine','Linear Regression']
Train_time = [
    train_time_dec,
    train_test_rand,
    train_test_est,
    train_test_ada,
    train_time_svr,
    train_time_linear    
]

index = np.arange(len(Model))
plt.bar(index, Train_time)
plt.xlabel("Machine Learning Models", fontsize =20)
plt.ylabel("Training Time", fontsize = 25)
plt.xticks(index, Model, fontsize=10)
plt.title("Comparision of Training Time taken of all the ML Models")
plt.show()

Model = ['Decision Tree','Random Forest','Gradiant Boosting','AdaBoost']
            
Train_time = [
    train_time_dec,
    train_test_rand,
    train_test_est,
    train_test_ada,
  
]

index = np.arange(len(Model))
plt.bar(index, Train_time)
plt.xlabel("Machine Learning Models", fontsize =20)
plt.ylabel("Training Time", fontsize = 25)
plt.xticks(index, Model, fontsize=10)
plt.title("Comparision of Training Time taken of all the ML Models")
plt.show()

# STEP 10 : Improve your accuracy basis K-Fold methods

# Gradiant Boosting Regressor
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = est, X = x_train, y=y_train, cv = 10)
accuracy

# STEP 11 : Conclusion

We have seen that accuracy of DecisionTree / Gradient Boosting is around 96.8% and 
also achieved decent variance score of 88.5% which is very close to 1.
Therefore, it is inferred that Gradient Boosting is the suitable model for this dataset.

** Thank You Very Much Indeed !!!


