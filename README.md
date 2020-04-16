# MMA_predictions
Using basic supervised learning to predict fight outcomes

The current dataset is restricted to the UFC

Have not fine tuned any of the models.
The Deep Learning models are WIP & not operational

The overall proecss of acquiring data and running the models needs to be automated.  

<b>To run:</b>

## Step 1
Create "fight_card.xlsx" in the data directory with the following structure
weight_class      fighter1    fighter2
or populate the 'df_fight_card' DataFrame
This contains the bouts for which outcomes would get predicted

## Step 2
#### If you want to get the latest fight data. 
#### Will take a while to run, since it gathers the  entire dataset.  Have not tailored  to only update data since last collection.
Execute src/get_fight_data.py  

## Step 3
Execute src/prep_test_data.py

## Step 4
Execute src/fight_predict.py

<i>Ed Bharucha</i> 