###############################################################################################
# Author: @ebharucha
# Date: 21/9/2019, 1/8/2020
###############################################################################################
import prep_test_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

prep_test_data.main()

def scale(scaler, data):
  if (scaler == 'standard'):
    std_scaler = StandardScaler()
    return (std_scaler.fit_transform(data))
  elif (scaler == 'minmax'):
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))
    return (minmax_scaler.fit_transform(data))

def main():
    # Load fight card
    fight_card_final = pd.read_csv('../data/fight_card_final.csv')
    df_fight_card = pd.read_excel('../data/fight_card.xlsx', sheet_name='Sheet1')

    # Load selected feature columns
    with open('../data/selected_features.pkl', 'rb') as pklfile:
        sel_features = pickle.load(pklfile)

    X_test = fight_card_final[sel_features]
    X_test = scale('standard', X_test)

    # Load classifer from pre-svaed model
    model = '../data/models/xgboost.pkl'
    with open(model, 'rb') as pklfile:
        classifier = pickle.load(pklfile)

    # Predict
    y_pred = classifier.predict(X_test)
    
    preds = []
    for idx, val in enumerate(y_pred):
        if (val == 0):
            preds.append(df_fight_card.fighter1[idx])
        else:
            preds.append(df_fight_card.fighter2[idx])

    df_fight_card['Winner Prediction'] = preds

    print (df_fight_card)


if __name__ == "__main__":
    main()