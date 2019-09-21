###############################################################################################
# Author: @ebharucha
# Date: 21/9/2019
###############################################################################################
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

def scale(scaler, data):
  if (scaler == 'standard'):
    std_scaler = StandardScaler()
    return (std_scaler.fit_transform(data))
  elif (scaler == 'minmax'):
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))
    return (minmax_scaler.fit_transform(data))

def main():
    # Load datasets
    with open ('../data/fight_data_final.pkl', 'rb') as pklfile:
        fight_data_final = pickle.load(pklfile)
    with open ('../data/fight_card_final.pkl', 'rb') as pklfile:
        fight_card_final = pickle.load(pklfile)
    df_fight_card = pd.read_excel('../data/fight_card.xlsx', sheet_name='Sheet1')

    # Shallow Learning
    # Prepare train & test datasets

    fight_data_final.drop(columns=['fighter1','fighter2', 'winner'], inplace=True)
    X = scale('standard', fight_data_final.iloc[:,:-1])
    y = fight_data_final.iloc[:,-1]
    X_test = scale('standard', fight_card_final)

    lgb_params = {
                 'n_estimators' : 500, 'boosting_type' : 'dart'
                 }

    lgbm = lgb.LGBMClassifier(**lgb_params)
    lgbm.fit(X, y)

    y_pred_lgbm = lgbm.predict(X_test)
    preds = []

    for idx, val in enumerate(y_pred_lgbm):
        if (val == 0):
            preds.append(df_fight_card.fighter1[idx])
        else:
            preds.append(df_fight_card.fighter2[idx])

    df_fight_card['Winner Prediction'] = preds

    print (df_fight_card)


if __name__ == "__main__":
    main()