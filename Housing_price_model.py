# Python Project Template

# 1. Prepare Problem
    # a) Load libraries
   
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

 # b) Load dataset
train = pd.read_csv("D:/Balaji/Kaggle/House price/train.csv")
test =  pd.read_csv("D:/Balaji/Kaggle/House price/test.csv")

# 2. Summarize Data
    # a) Descriptive statistics
    # b) Data visualizations
    
# shape & data types
print(train.shape[0])
print(train.dtypes)

# descriptions
pd.set_option('precision', 1)
print(train.describe())
    
# correlation
pd.set_option('precision', 2)
print(train.corr(method='pearson'))

# histograms
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
train.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()

#density plots
train.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=True,fontsize=4)
plt.show()

# box and whisker plots
train.plot(kind='box', subplots=True, layout=(8,8), sharex=False, sharey=False,fontsize=8)
plt.show()


# correlation matrix
 
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(train.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,81,10)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
plt.show()

#Log transform target and check the histogram
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()

# Get all features except Id and SalePrice
attr = train.columns.difference(['Id','SalePrice'])

# combine train and test data attributes for data pre-processing
all_data = pd.concat((train.loc[:,attr],test.loc[:,attr]))
         
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])


#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Dummy the categorical data in the dataframe
all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

# create combined first and second floor square foot based on sarthak suggestion in Kaggle
#Thanks Sarthak for your valuable hint
all_data['1stFlr_2ndFlr_Sf'] = np.log1p(all_data['1stFlrSF'] + all_data['2ndFlrSF'])

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# Create cross validation RMSE metric. Thanks to Alexandry post
from sklearn.model_selection import cross_val_score

#def rmse_cv(model):
#    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
#    return(rmse)
    
# Test options and evaluation metric
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'mean_squared_error'
    
# Spot-Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold=cross_validation.KFold(n=num_instances,n_folds=num_folds,random_state=seed)
 cv_results = np.sqrt(-cross_validation.cross_val_score(model, X_train, y, cv=kfold,scoring=scoring))
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',
Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
 kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds,random_state=seed)
 cv_results = np.sqrt(-cross_validation.cross_val_score(model, X_train, y, cv=kfold,scoring=scoring))
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Adding Xgboost model
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
 #the params were tuned using xgb.cv
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

#weighting ensemble model
preds = 0.7*lasso_preds + 0.3*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("ridge_sol.csv", index = False)

