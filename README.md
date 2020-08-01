# MMA_predictions
Using basic supervised learning to predict fight outcomes.

The current dataset is restricted to the UFC.

The current model is based on gradient boosting.  The Deep Learning models are WIP & not operational

The overall proecss of acquiring data and running the models needs to be automated.  

I also plan to introduce a basic web interface.

**To run:**

## Step 1
Download the repo

## Step 2
Create "fight_card.xlsx" in the data directory with the following structure<br>
e.g.<br>
**weight_class	fighter1	fighter2**<br>
Middleweight	Edmen Shahbazyan	Derek Brunson<br>
Women's Flyweight	Jennifer Maia	Joanne Calderwood<br>
Welterweight	Randy Brown	Vicente Luque<br>
Lightweight	Justin Gaethje	Khabib Nurmagomedov<br>
<br>
*This contains the bouts for which outcomes would get predicted*<br>

## Step 3
Execuete the following commands (without the quotes)<br>
"cd src"<br>
"python src/fight_predict.py"<br>


**Note:**<br>
If you want to get the latest fight data, execute ***"src/get_fight_data.py"**<br>
*Will take a while to run, since it gathers the  entire dataset.  Have not tailored  to only update data since last collection.*<br>
<br>
**@ebharucha**
