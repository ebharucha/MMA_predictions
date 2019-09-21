
###############################################################################################
# Author: @ebharucha
# Date: 7/9/2019
###############################################################################################
import numpy as np
import pandas as pd
import pickle
import category_encoders as ce
import warnings

warnings.filterwarnings("ignore")

def main():
    data_dir = '../data'
    # Load fight, fighter data
    with open(f'{data_dir}/fighter_data_en.pkl', 'rb') as pklfile:
        fighter_data_en = pickle.load(pklfile)

    # Load fight card for prediction
    df_fight_card = pd.read_excel('../data/fight_card.xlsx', sheet_name='Sheet1')
    
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

    fight_card_final.to_csv(f'{data_dir}/fight_card_final.csv')
    with open (f'{data_dir}/fight_card_final.pkl', 'wb') as pklfile:
        pickle.dump(fight_card_final, pklfile)


if __name__ == "__main__":
    main()
