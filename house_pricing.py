import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, norm, probplot, boxcox
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFECV, SelectFromModel

# read the train and test dataset's
train = pd.read_csv(r'G:\ml\house-prices-advanced-regression-techniques\train.csv')
test = pd.read_csv(r'G:\ml\house-prices-advanced-regression-techniques\test.csv')
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

train.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)
test.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)
train.rename(columns={'1stFlrSF':'FstFlrSF'}, inplace=True)
test.rename(columns={'1stFlrSF':'FstFlrSF'}, inplace=True)

test['SalePrice'] = 0


def rstr(df, pred=None): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum()/ obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'uniques', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis], axis = 1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
        corr_col = 'corr '  + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
    
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str

details = rstr(train, 'SalePrice')
display(details.sort_values(by='corr SalePrice', ascending=False))


# dropping outliers
train = train.drop(train[(train.GrLivArea>4000) & (train.SalePrice<300000)].index)
train = train[train.GrLivArea * train.TotRmsAbvGrd < 45000]
train = train[train.GarageArea * train.GarageCars < 3700]
train = train[(train.FullBath + (train.HalfBath*0.5) + train.BsmtFullBath + (train.BsmtHalfBath*0.5))<5]
train = train.loc[~(train.SalePrice==392500.0)]
train = train.loc[~((train.SalePrice==275000.0) & (train.Neighborhood=='Crawfor'))]
train = train.drop(train[(train.TotalBsmtSF>4000)].index)
train = train.drop(train[(train.FstFlrSF>2900)].index)
train.shape


train.drop(['Alley','GarageYrBlt','PoolQC','Fence', 'MiscFeature','Utilities'],axis=1,inplace=True)

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])
train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])

train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])

train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train.shape



test.drop(['Alley','GarageYrBlt','PoolQC','Fence', 'MiscFeature','Utilities'],axis=1,inplace=True)

#print(test.columns)
test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
test['FireplaceQu'] = test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])
test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])

test['GarageFinish'] = test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])
test['GarageQual'] = test['GarageQual'].fillna(test['GarageQual'].mode()[0])
test['GarageCond'] = test['GarageCond'].fillna(test['GarageCond'].mode()[0])
test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
#test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mode()[0])
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mode()[0])
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mode()[0])
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mode()[0])
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mode()[0])
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])



final_df = pd.concat([train,test],axis=0)
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

def categorical_onehot_multicols(multcols):
    df_final = final_df
    i=0
    for fields in multcols:
        print(fields)
        f1 = pd.get_dummies(final_df[fields],drop_first=True)
        final_df.drop([fields],axis=1,inplace =True)
        if i==0:
            df_final = f1.copy()
        else :
            df_final = pd.concat([df_final,f1],axis=1)
        i=i+1
    df_final = pd.concat([final_df,df_final],axis=1)
    return df_final

final_df = categorical_onehot_multicols(qualitative)
final_df = final_df.loc[:,~final_df.columns.duplicated()]        
final_df
Train = final_df.iloc[:1448,:]
Test = final_df.iloc[1448:,:]
Test= Test.drop(['SalePrice'],axis=1)
y_train = Train['SalePrice']
x_train = Train.drop(['SalePrice'],axis=1)
x_train.shape
y_train.head(5)
y_train = np.log1p(y_train)
back = np.expm1(y_train)
y_train.head(5)
back.head(5)
import xgboost
from sklearn.model_selection import RandomizedSearchCV

regressor = xgboost.XGBRegressor()
booster = ['gbtree','gblinear']
bas_score = [0.1,0.15,0.2,0.25,0.5,0.75,1]
n_estimators = [100,500,800,850,900,950,1100]
max_depth = [2,3,4,5,7,10]
learning_rate =[0.001,0.01,0.05,0.1,0.15,0.2]
min_child_weight=[1,2,3,4,5]

hyperparameter_grid = {'n_estimators':n_estimators,'max_depth':max_depth,'learning_rate':learning_rate,'min_child_weight':min_child_weight,'booster':booster,'base_score':bas_score}
random_cv = RandomizedSearchCV(estimator=regressor,param_distributions=hyperparameter_grid,cv=5,n_iter=50,scoring = 'neg_mean_absolute_error',n_jobs=4,verbose=5,return_train_score=True,random_state=30)
random_cv.fit(x_train,y_train)

random_cv.best_estimator_

regressor = xgboost.XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=950,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

regressor.fit(x_train,y_train)

predict = regressor.predict(Test)
predict
predict= np.expm1(predict)
print(Test.shape)
pred = pd.DataFrame(predict)
print(pred.head())
sub_df = pd.read_csv(r'G:\ml\house-prices-advanced-regression-techniques\sample_submission.csv')
data = pd.concat([sub_df['Id'],pred],axis=1)
data.columns = ['Id','SalePrice']
#data['Id'] = data['Id'].apply(np.int64)
data.tail()
data.to_csv(r'C:\Users\admin\Desktop\sample_submission.csv',index=False)
