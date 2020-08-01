###############################################################################################
# Author: @ebharucha
# Date: 1/8/2020
###############################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Univariate feature selection
def univariate_feature_sel(X, y):
    threshold = 5
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    sel_features = featureScores[featureScores.Score >= threshold].sort_values(by=['Score'], ascending=False).Specs
    return (sel_features)

# Function to scale data
def scale(scaler, data):
  if (scaler == 'standard'):
    std_scaler = StandardScaler()
    return (std_scaler.fit_transform(data))
  elif (scaler == 'minmax'):
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))
    return (minmax_scaler.fit_transform(data))

# Function to perform hyperparameter optimization via RandomizedSearchCV
def tune_classifier(model, nbr_iter, X, y):
    if (model == 'lgbm'):
        classifier = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', n_jobs=-1)
        p_dist = {'boosting_type': ['gbdt', 'dart'],
                'num_leaves': np.arange(2,260,2),
                'max_bin': np.arange(50,150,10),
                'n_estimators': np.arange(10, 40),
                'learning_rate': np.arange(0.01, 0.11, 0.01),
                }
    elif (model == 'xgboost'):
        classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='error', n_jobs=-1)
        p_dist = {'eta': np.arange(0.01,0.31, 0.01),
                'max_depth': np.arange(3, 11),
                'subsample': np.arange(0.5, 1.01, 0.1),
                'colsample_bytree': np.arange(0.5, 1.01, 0.1),
        }
    elif (model == 'rf'):
        classifier = RandomForestClassifier(n_jobs=-1)
        p_dist={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,300,400,500],
              'max_features':randint(1,31),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,4),
              }
    elif (model == 'knn'):
        classifier = KNeighborsClassifier(n_jobs=-1)
        p_dist={'leaf_size':np.arange(1,50),
              'n_neighbors':np.arange(1,30),
              'p':[1,2],
              }

    rdmsearch = RandomizedSearchCV(classifier, param_distributions=p_dist,\
         n_jobs=-1, n_iter=nbr_iter, cv=10)  
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return (ht_params, ht_score)


# Function to initialize Classifier depending on chosen model
def choose_classifier(model):
    if (model == 'lgbm'):
        lgb_params = {'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': 150,
                    'n_estimators': 28,
                    'max_bin': 90,
                    'learning_rate': 0.01,
                    'boosting_type': 'gbdt',
                    'n_jobs': -1}
        classifier = lgb.LGBMClassifier(**lgb_params)
    elif (model == 'xgboost'):
        xgb_params = {'objective': 'binary:logistic',
                    'eval_metric': 'error',
                    'subsample': 0.6,
                    'max_depth': 3,
                    'eta': 0.14,
                    'colsample_bytree': 1.0,
                    'n_jobs': -1}
        classifier = xgb.XGBClassifier(**xgb_params)
    elif (model == 'rf'):
        rf_params = {'bootstrap': True,
                    'criterion': 'entropy',
                    'max_depth': 10,
                    'max_features': 10,
                    'min_samples_leaf': 2,
                    'n_estimators': 500}
        classifier = RandomForestClassifier(**rf_params)
    elif (model == 'knn'):
        knn_params = {'p': 1,
                    'n_neighbors': 29,
                    'leaf_size': 10}
        classifier = KNeighborsClassifier(**knn_params)
    return (classifier)

# Function to perform StratifiedKFold validation
def strat_k_fold(classifier, X, y, n_splits):
    accuracy = []
    skf = StratifiedKFold(n_splits=n_splits, random_state=None)
    skf.get_n_splits(X, y)
    for train_index, test_index in skf.split(X,y):
        X1_train, X1_test = X[train_index], X[test_index]
        y1_train, y1_test = y.iloc[train_index], y.iloc[test_index]
        classifier.fit(X1_train, y1_train)
        pred = classifier.predict(X1_test)
        score = accuracy_score(pred, y1_test)
        accuracy.append(score)
    return(accuracy)

def main():
    # Load dataset
    fight_data_final =  pd.read_csv('../data/fight_data_final.csv')
    fight_data_final.drop(columns=['fighter1', 'fighter2', 'winner', 'Unnamed: 0'], inplace=True)
    # Split into feature & target columns
    X = fight_data_final.iloc[:,:-1]
    y = fight_data_final.iloc[:, -1]
    # User Univariate analysis to narrow down features
    sel_features = univariate_feature_sel(X, y)
    X = X[sel_features]
    # Scale data
    # X_scaled = scale('minmax', X)
    X_scaled = scale('standard', X)
    # Store selected features for futrue use
    with open('../data/selected_features.pkl', 'wb') as pklfile:
        pickle.dump(sel_features, pklfile)

    # Define list of models to evaluate
    models = ['xgboost', 'lgbm', 'rf', 'knn']
    # models = ['xgboost']

    # Create directory to store models if it doesn't exist
    model_dir = '../data/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Evaluate each model
    for model in models:
        # Hyperparameter tuning using RandomizedSearchCV
        # (ht_params, ht_score) = tune_classifier(model, 50, X_scaled, y)
        # print(model, ht_params, ht_score)
        # Reconfirm model performance after hyperparameter tuning above
        classifier = choose_classifier(model)
        score = cross_val_score(classifier, X_scaled, y, cv=10) # KFold cross validation
        accuracy = strat_k_fold(classifier, X_scaled, y, 10)
        print (f'{model}: KFold={score.mean()*100:.2f}%, StratifiedKFold={np.array(accuracy).mean()*100:.2f}%')
        # Save model
        with open(f'{model_dir}/{model}.pkl', 'wb') as pklfile:
            pickle.dump(classifier, pklfile)

if __name__ == "__main__":
    main()