
Explore an Environment
-----------------
In this task, new cells can grow only next to already
living cells. Each cell has an energy value that controls its fire rate. 
The goal is, starting from a single alive cell, to find a randomly placed 
target while using in total the lowest amount of energy.

------------------

The neural CA presented in `CreaterEnergyBudget` (also the pretrained model) has an energy budget of 100 in the 
compared to 20 for the one in this folder.

In the two videos in `CreaterEnergyBudget`, the same model is used, only the gird size changes.
We can observe the formation of a structure around the input and target cell once found.
This seems to be to stop the propagation of living gliders around the structure. 


In the first video, with an energy budget of 20, the cells are not able to fill
the space as they do with 100. They stay at the limit of being alive with energy
values ~ 0.1. The exploration is thus much slower and limited in width but uses less
energy.