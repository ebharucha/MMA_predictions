###############################################################################################
# Author: @ebharucha
# Date: 7/9/2019
###############################################################################################
import requests
from bs4 import BeautifulSoup
import string
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import re
import category_encoders as ce
import warnings

warnings.filterwarnings("ignore")

###############################################################################################
def height_cm(height):
    try:
        feet = height.split('"')[0].split("'")[0]
    except:
        feet = ''
    try:
        inches = height.split('"')[0].split("'")[1]
    except:
        inches = ''

    if not feet.strip().isdigit():
        feet = 0
    if not inches.strip().isdigit():
        inches = 0
    return (int(feet)*30.48 + int(inches)*2.54)
###############################################################################################
def reach_cm(reach):
    regnumber = re.compile(r'\d+(?:,\d*)?')
    rch = reach.split('"')[0]

    if regnumber.match(rch.strip()):
        inches = rch.split('.')[0]
    else:
        inches = 0
    
    return(int(inches)*2.54)
###############################################################################################
def weight_lbs(weight):
    lbs = weight.split(' ')[0]
    if not lbs.strip().isdigit():
        lbs = 0
    
    return(int(lbs))
###############################################################################################
def get_fighter_data(data_dir):
    first_names = []
    last_names = []
    nick_names = []
    heights = []
    weights = []
    reaches = []
    stances = []
    wins = []
    losses = []
    draws = []
    fighter_urls = []

    for c in list(string.ascii_lowercase):
        fighters_url = f'http://ufcstats.com/statistics/fighters?char={c}&page=all'
        
        page = requests.get(fighters_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        fighters_data = soup.find_all('tr')

        for fighter in tqdm(fighters_data[2:]):
            try:
                fighter_data = fighter.find_all('td')
                first_names.append(fighter_data[0].get_text(strip=True))
                last_names.append(fighter_data[1].get_text(strip=True))
                nick_names.append(fighter_data[2].get_text(strip=True))
                heights.append(fighter_data[3].get_text(strip=True))
                weights.append(fighter_data[4].get_text(strip=True))
                reaches.append(fighter_data[5].get_text(strip=True))
                stances.append(fighter_data[6].get_text(strip=True))
                wins.append(fighter_data[7].get_text(strip=True))
                losses.append(fighter_data[8].get_text(strip=True))
                draws.append(fighter_data[9].get_text(strip=True))
                fighter_urls.append(fighter_data[0].find('a')['href'])
            except:
                pass

    fighter_data = pd.DataFrame(
        {
            'first_name' : first_names,
            'last_name' : last_names,
            'nick_name' : nick_names,
            'height' : heights,
            'weight' : weights,
            'reach' : reaches,
            'stance' : stances,
            'wins' : wins,
            'losses' : losses,
            'draws' : draws,
            'fighter_url' : fighter_urls,
        }
    )

    SLpM = []
    Str_Acc = []
    SApM = []
    Str_Dep = []
    TD_Avg = []
    TD_Acc = []
    TD_Def = []
    Sub_Avg = []

    for url in tqdm(fighter_data.fighter_url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        try:
            career_stats = soup.find(class_='b-list__info-box b-list__info-box_style_middle-width js-guide clearfix').find_all('li')
        except:
            pass

        SLpM.append(career_stats[0].get_text(strip=True).split(':')[1])
        Str_Acc.append(career_stats[1].get_text(strip=True).split(':')[1])
        SApM.append(career_stats[2].get_text(strip=True).split(':')[1])
        Str_Dep.append(career_stats[3].get_text(strip=True).split(':')[1])
        TD_Avg.append(career_stats[5].get_text(strip=True).split(':')[1])
        TD_Acc.append(career_stats[6].get_text(strip=True).split(':')[1])
        TD_Def.append(career_stats[7].get_text(strip=True).split(':')[1])
        Sub_Avg.append(career_stats[8].get_text(strip=True).split(':')[1])

    fighter_data['SLpM'] = SLpM
    fighter_data['Str_Acc'] = Str_Acc
    fighter_data['SApM'] = SApM
    fighter_data['Str_Dep'] = Str_Dep
    fighter_data['TD_Avg'] = TD_Avg
    fighter_data['TD_Acc'] = TD_Acc
    fighter_data['TD_Def'] = TD_Def
    fighter_data['Sub_Avg'] = Sub_Avg

    fighter_data['full_name'] = fighter_data.first_name + " " + fighter_data.last_name

    fighter_data.to_csv(f'{data_dir}/fighter_data.csv') 

    return (fighter_data)
###############################################################################################
def encode_fighter_data(data_dir, fighter_data):
    fighter_data['height'] = fighter_data['height'].apply(lambda x: height_cm(x))
    fighter_data['reach'] = fighter_data['reach'].apply(lambda x: reach_cm(x))
    fighter_data['weight'] = fighter_data['weight'].apply(lambda x: weight_lbs(x))

    fighter_data.wins = pd.to_numeric(fighter_data.wins)
    fighter_data.losses = pd.to_numeric(fighter_data.losses)
    fighter_data.draws = pd.to_numeric(fighter_data.draws)

    enc = ce.HashingEncoder(cols = ['stance'])
    fighter_data = enc.fit_transform(fighter_data)

    fighter_data['SLpM'] = pd.to_numeric(fighter_data['SLpM'])

    fighter_data['Str_Acc'] = fighter_data['Str_Acc'].str.replace('%', '')
    fighter_data['Str_Acc'] = pd.to_numeric(fighter_data['Str_Acc'])

    fighter_data['SApM'] = pd.to_numeric(fighter_data['SApM'])

    fighter_data['Str_Dep'] = fighter_data['Str_Dep'].str.replace('%', '')
    fighter_data['Str_Dep'] = pd.to_numeric(fighter_data['Str_Dep'])

    fighter_data['TD_Avg'] = pd.to_numeric(fighter_data['TD_Avg'])

    fighter_data['TD_Acc'] = fighter_data['TD_Acc'].str.replace('%', '')
    fighter_data['TD_Acc'] = pd.to_numeric(fighter_data['TD_Acc'])

    fighter_data['TD_Def'] = fighter_data['TD_Def'].str.replace('%', '')
    fighter_data['TD_Def'] = pd.to_numeric(fighter_data['TD_Def'])

    fighter_data['Sub_Avg'] = pd.to_numeric(fighter_data['Sub_Avg'])


    fighter_data.to_csv(f'{data_dir}/fighter_data_en.csv')
    with open (f'{data_dir}/fighter_data_en.pkl', 'wb') as pklfile:
        pickle.dump(fighter_data, pklfile)

    return (fighter_data)
###############################################################################################
def get_fight_data(data_dir):
    fight_data_url = 'http://ufcstats.com/statistics/events/completed?page=all'

    page = requests.get(fight_data_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    fight_list = soup.find('tbody').find_all('a')
    fight_card_names = [fight.get_text(strip=True) for fight in fight_list]
    fight_card_urls = [fight['href'] for fight in fight_list]

    fight_card = []
    fight_card_url = []
    fighter1 = []
    fighter2 = []
    weight_class = []
    winner = []
    method = []
    round_ = []
    i = 0

    for url in tqdm(fight_card_urls[1:]):
        page = requests.get(url)
        soup = BeautifulSoup(page.content,'html.parser')
        full_fight_card = soup.find('tbody')
        full_fight_card_fights = full_fight_card.find_all('tr')
        for fight in full_fight_card_fights:
            fight_card.append(fight_card_names[i+1])
            fight_card_url.append(url)
            fighter1.append(fight.find_all('td')[1].find_all('a')[0].get_text(strip=True))
            fighter2.append(fight.find_all('td')[1].find_all('a')[1].get_text(strip=True))
            weight_class.append(fight.find_all('td')[6].get_text(strip=True))
            winner.append(fighter1[-1])
            method.append(fight.find_all('td')[7].get_text(strip=True))
            round_.append(fight.find_all('td')[8].get_text(strip=True))
        i+=1

    fight_data = pd.DataFrame(
    {
        'fight_card' : fight_card,
        'fight_card_url' : fight_card_url,
        'fighter1' : fighter1,
        'fighter2' : fighter2,
        'weight_class' : weight_class,
        'winner' : winner,
        'method' : method,
        'round' : round_,
    }
    )

    # Shuffle winner netween fighter1 & fighter2
    start_idx = 25
    inc = 25
    end_idx = start_idx + inc

    while start_idx <= fight_data.shape[0]:
        fight_data.loc[start_idx:end_idx,['fighter1', 'fighter2']] = fight_data.loc[start_idx:end_idx,['fighter2', 'fighter1']].values
        start_idx = end_idx + 25
        end_idx = start_idx + inc


    fight_data.to_csv(f'{data_dir}/fight_data.csv')
    with open (f'{data_dir}/fight_data.pkl', 'wb') as pklfile:
        pickle.dump(fight_data, pklfile)

    return(fight_data)
###############################################################################################
def prep_fight_data_final(data_dir, fighter_data_en, fight_data_relevant):
    # Get data for fighter 1
    fight_data_relevant = fight_data_relevant.merge(fighter_data_en, left_on=['fighter1'], right_on=['full_name'],\
                                                    how='left').drop(columns=['full_name'])
    fight_data_relevant.rename(columns={'col_0':'stance_col_0_fighter1', 'col_1':'stance_col_1_fighter1', 'col_2':'stance_col_2_fighter1',\
                                        'col_3':'stance_col_3_fighter1', 'col_4':'stance_col_4_fighter1', 'col_5':'stance_col_5_fighter1',\
                                        'col_6':'stance_col_6_fighter1', 'col_7':'stance_col_7_fighter1',\
                                        'height':'height_fighter1', 'weight':'weight_fighter1', 'reach':'reach_fighter1',\
                                        'wins':'wins_fighter1', 'losses':'losses_fighter1', 'draws':'draws_fighter1',\
                                        'SLpM':'SLpM_fighter1','Str_Acc':'Str_Acc_fighter1', 'SApM':'SApM_fighter1',\
                                        'Str_Dep':'Str_Dep_fighter1', 'TD_Avg':'TD_Avg_fighter1', 'TD_Acc':'TD_Acc_fighter1',\
                                        'TD_Def':'TD_Def_fighter1', 'Sub_Avg':'Sub_Avg_fighter1'}, inplace=True)

    # Get data for fighter 2
    fight_data_relevant = fight_data_relevant.merge(fighter_data_en, left_on=['fighter2'], right_on=['full_name'],\
                                                    how='left').drop(columns=['full_name'])
    fight_data_relevant.rename(columns={'col_0':'stance_col_0_fighter2', 'col_1':'stance_col_1_fighter2', 'col_2':'stance_col_2_fighter2',\
                                        'col_3':'stance_col_3_fighter2', 'col_4':'stance_col_4_fighter2', 'col_5':'stance_col_5_fighter2',\
                                        'col_6':'stance_col_6_fighter2', 'col_7':'stance_col_7_fighter2',\
                                        'height':'height_fighter2', 'weight':'weight_fighter2', 'reach':'reach_fighter2',\
                                        'wins':'wins_fighter2', 'losses':'losses_fighter2', 'draws':'draws_fighter2',\
                                        'SLpM':'SLpM_fighter2','Str_Acc':'Str_Acc_fighter2', 'SApM':'SApM_fighter2',\
                                        'Str_Dep':'Str_Dep_fighter2', 'TD_Avg':'TD_Avg_fighter2', 'TD_Acc':'TD_Acc_fighter2',\
                                        'TD_Def':'TD_Def_fighter2', 'Sub_Avg':'Sub_Avg_fighter2'}, inplace=True)                                   

    # Encode weight class
    enc = ce.HashingEncoder(cols = ['weight_class'])
    fight_data_relevant = enc.fit_transform(fight_data_relevant)
    fight_data_relevant.rename(columns={'col_0':'weight_class_col_0', 'col_1':'weight_class_col_1', 'col_2':'weight_class_col_2',\
                                        'col_3':'weight_class_col_3', 'col_4':'weight_class_col_4', 'col_5':'weight_class_col_5',\
                                        'col_6':'weight_class_col_6', 'col_7':'weight_class_col_7'}, inplace=True)

    # Encode winner
    fight_data_relevant['winner_en'] = np.where(fight_data_relevant['winner']==fight_data_relevant['fighter1'], 0, 1)
    # fight_data_relevant.drop(columns=['fighter1', 'fighter2', 'winner'], inplace=True)

    fight_data_relevant.to_csv(f'{data_dir}/fight_data_final.csv')
    with open (f'{data_dir}/fight_data_final.pkl', 'wb') as pklfile:
        pickle.dump(fight_data_relevant, pklfile)

    return(fight_data_relevant)
###############################################################################################
def main():
    data_dir = '../data'
    fighter_data_relevant_columns = ['full_name','height','weight','reach','stance','wins','losses','draws',\
        'SLpM','Str_Acc','SApM','Str_Dep','TD_Avg','TD_Acc','TD_Def','Sub_Avg']
    fight_data_relevant_columns = ['fighter1', 'fighter2', 'weight_class', 'winner']

    fighter_data = get_fighter_data(data_dir)
    fighter_data_en = encode_fighter_data(data_dir, fighter_data[fighter_data_relevant_columns])

    fight_data = get_fight_data(data_dir)

    fight_data_relevant = fight_data[fight_data_relevant_columns]
    fight_data_final = prep_fight_data_final(data_dir, fighter_data_en, fight_data_relevant)
###############################################################################################
if __name__ == "__main__":
    main()