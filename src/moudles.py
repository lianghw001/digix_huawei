import pandas as pd
import numpy as np
import lightgbm as lgb
import progressbar
from scipy import sparse
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

def  useModel(df_train,df_test):
    
    print("训练模型：")
    param = {
            'learning_rate': 0.2,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 20,
            'objective': 'multiclass',
            'num_class': 7,
            'min_data_in_leaf': 50,
            'max_bin': 230,
            'metric': 'multi_error',
            'is_unbalance' :False,
            'bagging_fraction' :0.1,
            'feature_fraction' :0.1
            }

    X = df_train.drop(['age_group','uid'], axis=1)
    y = df_train['age_group']
    uid = df_test['uid']
    test = df_test.drop('uid', axis=1)

    xx_score = []
    cv_pred = []
    skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=True)
    for index, (train_index, vali_index) in enumerate(skf.split(X, y)):
        print(index)
        x_train, y_train, x_vali, y_vali = np.array(X)[train_index],\
         np.array(y)[train_index], np.array(X)[vali_index], np.array(y)[vali_index]
        train = lgb.Dataset(x_train, y_train)
        vali =lgb.Dataset(x_vali, y_vali)
        print("training start...")
        model = lgb.train(param, train, num_boost_round=1000, valid_sets=[vali], early_stopping_rounds=60)

        xx_pred = model.predict(x_vali,num_iteration=model.best_iteration)
        xx_pred = [np.argmax(x) for x in xx_pred]
        xx_score.append(f1_score(y_vali,xx_pred,average='weighted'))

        y_test = model.predict(test,num_iteration=model.best_iteration)
        y_test = [np.argmax(x) for x in y_test]
        
        if index == 0:
            cv_pred = np.array(y_test).reshape(-1, 1)
        else:
            cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
            
    submit = []
    for line in cv_pred:
        #投票
        submit.append(np.argmax(np.bincount(line)))
    df = pd.DataFrame({'id':uid.as_matrix(),'label':submit})

    return df

