import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lightgbm as lgb

def scale(scaler, data):
  if (scaler == 'standard'):
    std_scaler = StandardScaler()
    return (std_scaler.fit_transform(data))
  elif (scaler == 'minmax'):
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))
    return (minmax_scaler.fit_transform(data))


def fightCard(fighter_data_en, fight_data):
    fighters = fighter_data_en.full_name.unique()
    fighters = np.insert(fighters, 0, '')
    weight_classes = fight_data.weight_class.unique()
    weight_classes = np.sort(np.insert(weight_classes, 0, ''))
    st.header('Fighter 1')
    fighter1 = st.selectbox('', fighters)
    st.header('Fighter 2')
    fighter2 = st.selectbox(' ', fighters)
    st.header('Weight Class')
    weight_class = st.selectbox(' ', weight_classes)
    
    return (fighter1, fighter2, weight_class)

def prep_fight_card(df_fight_card, fighter_data_en):
    # Encode weight class
    enc = ce.HashingEncoder(cols = ['weight_class'])
    fight_card_final = enc.fit_transform(df_fight_card)
    fight_card_final.rename(columns={'col_0':'weight_class_col_0', 'col_1':'weight_class_col_1', 'col_2':'weight_class_col_2',\
                                        'col_3':'weight_class_col_3', 'col_4':'weight_class_col_4', 'col_5':'weight_class_col_5',\
                                        'col_6':'weight_class_col_6', 'col_7':'weight_class_col_7'}, inplace=True)

    # Get data for fighter1
    fight_card_final = fight_card_final.merge(fighter_data_en, left_on=['fighter1'], right_on=['full_name'],\
                                                    how='left').drop(columns=['full_name', 'fighter1'])
    fight_card_final.rename(columns={'col_0':'stance_col_0_fighter1', 'col_1':'stance_col_1_fighter1', 'col_2':'stance_col_2_fighter1',\
                                        'col_3':'stance_col_3_fighter1', 'col_4':'stance_col_4_fighter1', 'col_5':'stance_col_5_fighter1',\
                                        'col_6':'stance_col_6_fighter1', 'col_7':'stance_col_7_fighter1',\
                                        'height':'height_fighter1', 'weight':'weight_fighter1', 'reach':'reach_fighter1',\
                                        'wins':'wins_fighter1', 'losses':'losses_fighter1', 'draws':'draws_fighter1',\
                                        'SLpM':'SLpM_fighter1','Str_Acc':'Str_Acc_fighter1', 'SApM':'SApM_fighter1',\
                                        'Str_Dep':'Str_Dep_fighter1', 'TD_Avg':'TD_Avg_fighter1', 'TD_Acc':'TD_Acc_fighter1',\
                                        'TD_Def':'TD_Def_fighter1', 'Sub_Avg':'Sub_Avg_fighter1'}, inplace=True)


    # Get data for fighter 2
    fight_card_final = fight_card_final.merge(fighter_data_en, left_on=['fighter2'], right_on=['full_name'],\
                                                    how='left').drop(columns=['full_name', 'fighter2'])
    fight_card_final.rename(columns={'col_0':'stance_col_0_fighter2', 'col_1':'stance_col_1_fighter2', 'col_2':'stance_col_2_fighter2',\
                                        'col_3':'stance_col_3_fighter2', 'col_4':'stance_col_4_fighter2', 'col_5':'stance_col_5_fighter2',\
                                        'col_6':'stance_col_6_fighter2', 'col_7':'stance_col_7_fighter2',\
                                        'height':'height_fighter2', 'weight':'weight_fighter2', 'reach':'reach_fighter2',\
                                        'wins':'wins_fighter2', 'losses':'losses_fighter2', 'draws':'draws_fighter2',\
                                        'SLpM':'SLpM_fighter2','Str_Acc':'Str_Acc_fighter2', 'SApM':'SApM_fighter2',\
                                        'Str_Dep':'Str_Dep_fighter2', 'TD_Avg':'TD_Avg_fighter2', 'TD_Acc':'TD_Acc_fighter2',\
                                        'TD_Def':'TD_Def_fighter2', 'Sub_Avg':'Sub_Avg_fighter2'}, inplace=True)

    return(fight_card_final)

def fight_pred(fight_card, fight_card_final, fight_data_final):
    fight_data_final.drop(columns=['Unnamed: 0', 'fighter1','fighter2', 'winner'], inplace=True)
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
            preds.append(fight_card.fighter1[idx])
        else:
            preds.append(fight_card.fighter2[idx])

    fight_card['Winner Prediction'] = preds
    return(fight_card)


def main():
    st.title('MMA Fight Predicions')
    
    fighter_data_en = pd.read_csv('./data/fighter_data_en.csv')
    fight_data = pd.read_csv('./data/fight_data.csv')
    fight_data_final = pd.read_csv('./data/fight_data_final.csv')
    (fighter1, fighter2, weight_class) = fightCard(fighter_data_en, fight_data)
    
    fighter1_ = []
    fighter2_ = []
    weight_class_ = []
    fight_card = pd.DataFrame()   
    if st.button('Add Fight'):
        fighter1_.append(fighter1)
        fighter2_.append(fighter2)
        weight_class_.append(weight_class)
        fight_card['weight_class'] = weight_class_
        fight_card['fighter1'] = fighter1_
        fight_card['fighter2'] = fighter2_
        fight_card_final = prep_fight_card(fight_card, fighter_data_en)
        fight_card = fight_pred(fight_card, fight_card_final, fight_data_final)

    st.table(fight_card)
    

if __name__ == "__main__":
    main()