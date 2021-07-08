Neural CA that compute boolean function with variable input and output cell position
------

We go further in exploring robustness abilities of neural CA by pertubing 
the position of the input and output cells. The input and output cells have
 their position randomized inside a circle of radius 2. We test it on the 
task of computing the non-linear boolean function NOT XOR.

Red values (+1) correspond to 1, blue values (-1) to 0.

The NOT XOR table becomes

     |  BLUE | RED
---------------------
BLUE |  RED  | BLUE
---------------------
RED  |  BLUE | RED

In the videos presented the NOT XOR computation with 3 differents input/output cells positions.
We observe the emergence of continuous gradient of values in the different channels around 
the input and outputs that could be interpreted as a way to locate these cells despites
random preturbation in their position.


The pretrained model has been trained with noise and damage even if it's not shown in 
the video.