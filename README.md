# MMA_predictions
Using basic supervised learning to predict fight outcomes

The current dataset is restricted to the UFC

Have not fine tuned any of the models.
The Deep Learning models are WIP & not operational

The overall proecss of acquiring data and running the models needs to be automated.  

**To run:**

## Step 1
Download the repo

## Step 2
Create "fight_card.xlsx" in the data directory with the following structure
e.g.
weight_class	fighter1	fighter2
Middleweight	Edmen Shahbazyan	Derek Brunson
Women's Flyweight	Jennifer Maia	Joanne Calderwood
Welterweight	Randy Brown	Vicente Luque
Lightweight	Justin Gaethje	Khabib Nurmagomedov

*This contains the bouts for which outcomes would get predicted*

## Step 3
Execute src/fight_predict.py


**__Note:__**.
*If you want to get the latest fight data.*
*Will take a while to run, since it gathers the  entire dataset.  Have not tailored  to only update data since last collection.*
*Execute src/get_fight_data.py*

**@ebharucha**
