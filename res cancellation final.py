# kaggle 
# reservation cancellation prediction 
# scoring : log_loss
# Algorithms: random forest, XGBoost, Light GBM, Cat Boost, Stack 
# which is the best?
# 27/02/2024

#final code

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

from sklearn.pipeline import Pipeline 

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier 
from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold, GridSearchCV

#Loading the data
os.chdir(r"C:\Neha\Sanes Academy Pune\Kaggle\Reservation cancellation")
train = pd.read_csv("train__dataset.csv")
test = pd.read_csv("test___dataset.csv")

#NA's --- no NA's found
print(train.isnull().sum())
print(test.isnull().sum())

#Categorical data : convert from int to str format for train & test set
train['type_of_meal_plan']=train['type_of_meal_plan'].astype(str)
train['room_type_reserved']=train['room_type_reserved'].astype(str)
train['market_segment_type']=train['market_segment_type'].astype(str)

test['type_of_meal_plan']=test['type_of_meal_plan'].astype(str)
test['room_type_reserved']=test['room_type_reserved'].astype(str)
test['market_segment_type']=test['market_segment_type'].astype(str)

#X, y, test_new
X = train.drop(['booking_status', 'arrival_date' ], axis=1)
y = train['booking_status']

test_new = test.drop('arrival_date', axis=1)

#X_train, X_test, y_train, y_test : train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, random_state=24)

#One hot encoder
ohe = OneHotEncoder(sparse_output=False, 
                    handle_unknown='ignore').set_output(transform='pandas')
col_trn = make_column_transformer(
    (ohe,make_column_selector(dtype_include=object)),
    ('passthrough', make_column_selector(dtype_exclude=object)),
    verbose_feature_names_out=False)
col_trn = col_trn.set_output(transform='pandas')

#Algorithms : random forest
rf = RandomForestClassifier(random_state=24)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(log_loss(y_test, y_pred))
log_loss_rf = log_loss(y_test, y_pred)

#XGBM
x_gbm = XGBClassifier(random_state=24)
# pipe
pipe = Pipeline([('TRNSR', col_trn), 
                ('XGBM', x_gbm)])
print(pipe.get_params())

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
params = { 
          # xgbm
          'XGBM__learning_rate':[0.1, 0.3, 0.6],
          'XGBM__max_depth':[None, 3, 5],
          'XGBM__n_estimators':[100, 50, 25],
        } 

gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_param_xgbm = gcv.best_estimator_
best_params_xgbm = gcv.best_params_
best_score_xgbm = gcv.best_score_

print(best_param_xgbm)
print(best_params_xgbm)
print(best_score_xgbm)

#LGBM
l_gbm =  LGBMClassifier(random_state=24)
# pipe
pipe = Pipeline([('TRNSR', col_trn), 
                ('LGBM', l_gbm)])
print(pipe.get_params())

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
params = { 
            # lgbm
          'LGBM__learning_rate':[0.1, 0.3, 0.6],
          'LGBM__max_depth':[None, 3, 5],
          'LGBM__n_estimators':[100, 50, 25]
        } 

gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_param_lgbm = gcv.best_estimator_
best_params_lgbm = gcv.best_params_
best_score_lgbm = gcv.best_score_
print(best_param_lgbm)
print(best_params_lgbm)
print(best_score_lgbm)

#Cat Boost
c_gbm =  CatBoostClassifier(random_state=24)
# pipe
pipe = Pipeline([('TRNSR', col_trn), 
                ('CGBM', c_gbm)])
print(pipe.get_params())

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
params = { 
            # cgbm
          'CGBM__learning_rate':[0.1, 0.3, 0.6],
          #'CGBM__max_depth':[None, 3, 5],
          'CGBM__n_estimators':[100, 50, 25]
        } 

gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')

gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_param_cgbm = gcv.best_estimator_
best_params_cgbm = gcv.best_params_
best_score_cgbm = gcv.best_score_
print(best_param_cgbm)
print(best_params_cgbm)
print(best_score_cgbm)

#Stack ensemble
l_gbm =  LGBMClassifier(random_state=24)
rf = RandomForestClassifier(random_state=24)
c_gbm =  CatBoostClassifier(random_state=24)
x_gbm = XGBClassifier(random_state=24)
stack = StackingClassifier([('RF', rf),
                 ('LGBM', l_gbm), ('CGBM', c_gbm)],
                           final_estimator=x_gbm, # x_gbm has the best score in the previous runs
                           passthrough=True) #, stack_method = 'predict_proba'
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)

pipe = Pipeline([('TRNSR', col_trn),  
                ('stack', stack )]) 
pipe.get_params()

params = {
          # xgbm
          'stack__final_estimator__learning_rate':[0.3, 0.5],
          'stack__final_estimator__max_depth':[None, 3],
          'stack__final_estimator__n_estimators':[100, 125],
          # lgbm
          'stack__LGBM__learning_rate':[0.3, 0.5],
          'stack__LGBM__max_depth':[None, 3],
          'stack__LGBM__n_estimators':[100, 125],
          # cgbm                  
          'stack__CGBM__learning_rate':[0.3, 0.5],
          #'stack__CGBM__depth':[None, 3, 5],
          'stack__CGBM__n_estimators':[100, 125]
          } 

gcv = GridSearchCV(pipe, param_grid=params,cv=kfold,
                   verbose=3,  scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_param_stack = gcv.best_estimator_
#y_pred_stack = best_param_stack.predict(y_test)

# IMP : Final stack results :{'stack__CGBM__learning_rate': 0.3, 'stack__CGBM__n_estimators': 125, 'stack__LGBM__learning_rate': 0.3, 'stack__LGBM__max_depth': None, 'stack__LGBM__n_estimators': 100, 'stack__final_estimator__learning_rate': 0.3, 'stack__final_estimator__max_depth': 3, 'stack__final_estimator__n_estimators': 100}
# -0.2513184632124158

best_param_stack = gcv.best_estimator_
best_params_stack = gcv.best_params_
best_score_stack = gcv.best_score_
print(best_param_stack)
print(best_params_stack)
print(best_score_stack)

#Compare the log_loss:
print("RF:")
print(log_loss_rf)
print("XGBM:")
print(best_param_xgbm)
print(best_params_xgbm)
print(best_score_xgbm)
print("LGBM:")
print(best_param_lgbm)
print(best_params_lgbm)
print(best_score_lgbm)
print("CGBM:")
print(best_param_cgbm)
print(best_params_cgbm)
print(best_score_cgbm)
print("Stack:")
print(best_param_stack)
print(best_params_stack)
print(best_score_stack)    

# in stages : {'stack__CGBM__learning_rate': 0.3, 'stack__CGBM__n_estimators': 100, 'stack__LGBM__learning_rate': 0.3, 'stack__LGBM__max_depth': None, 'stack__LGBM__n_estimators': 100, 'stack__final_estimator__learning_rate': 0.3, 'stack__final_estimator__max_depth': None, 'stack__final_estimator__n_estimators': 100}
# -0.2664657704425317

#############################################################################################
# streamlit app : how to use video for best model
# https://www.youtube.com/watch?v=mdgfwnesAdI
from joblib import dump

# getting the best model object
best_model = best_param_stack

# serializing the best model object into a file
dump(best_model, 'stack-best.joblib')

###########################################################################
# is this supervised learning?
# which type of supervised learning is this?
# Ans : supervised learning - classification
