# Towards self-organized control
We used neural cellular automata to robustly control a 
cart-pole agent.

This repository host the interactive article [*Towards self-organized control*](https://avariengien.github.io/self-organized-control/)
as well as the code and a [Google Colab notebook](https://colab.research.google.com/github/aVariengien/self-organized-control/blob/main/code/Towards-self-organized-control-notebook.ipynb) 
to easily  reproduce the results and experiment
with the pretrained models.

The [paper](https://www.crosslabs.org/articles/imi-2021-01) was published in the Innovations in Machine Intelligence (IMI) journal.


### Structure

In the `code` folder:
 * The `SelfOrgControl` package that host the class and the function to build and run the neural CA. You can install it with `pip install git+https://github.com/aVariengien/self-organized-control.git#subdirectory=code`
 * The `AdditionalExperiments` contains code and videos about other experiment with neural CA. Each experiment has its own `README.md` file.
 * The notebook `Towards-self-organized-control-notebook.ipynb`
 * The `demo` contains the javascript code used for the interactive demo using tensorflow.js
 
### Cite this article:

A. Variengien, S. Pontes-Filho, T. E. Glover, S. Nichele, "Towards Self-organized Control: Using Neural Cellular Automata to Robustly Control a Cart-pole Agent", Innovations in Machine Intelligence (IMI), vol. 1, pp. 1-14, 2021. DOI: 10.54854/imi2021.01
