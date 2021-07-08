import os

from SelfOrgControl.NeuralCA import *
from SelfOrgControl.AdditionalExpUtils import *

import random
import numpy as np
import tensorflow as tf
from collections import deque




## Initilize the model

# min and max number of CA update steps for a training step
MIN_STEP=30
MAX_STEP=50

# Here not all cells are alive, new cells can grow only close to already living cells.
# A cell becomes alive if its channel 1 (the energy channel) is > 0.1

# The energy channel is also controlling the firerate of each cell: the more
# energy a cell have, the more frequent is will be updated
nca = NeuralCA(use_per_cell_fire_rate=True, input_electrodes = [(16,16)],
                 output_electrodes = [(17,16)], output_visible=True,output_alive=False,
                 custom_input_signal=False,batch_size=8, size=32, channel_n=4,
                 fire_rate=0.9, first_lay_size= 10,nn_init="random", bound_values=False,
                 value_bounds=(-2,2), all_cells_alive=False, custom_channel_nb = 0,
                 enforce_connexity=False, energy_as_amplitude=False, max_amplitude=None,
                 bound_energy_change=False, torus_boundaries=False, add_noise=False,
                 use_hidden_inputs=False)


ENERGY_BUDGET = 100 #depending on the energy budget, different strategy will emerge
N_EPOCH = 500
pool = BatchPool(20, nca.size, nca.channel_n,
                nca.in_cells, nca.out_cells,  radius=15,
                batch_size=nca.batch_size, range_nb_hidden_cells=(0,0),
                damage=False, proba_move=1., move_inputs=False,
                move_outputs=True, init_type="zeros")

lr = 1e-3
trainer = tf.keras.optimizers.Adam(learning_rate=lr)



for i in range(N_EPOCH):

    if np.random.random() <0.9:
      x, out_coo, in_coo, hid = pool.sample()
    else:
      x, out_coo, in_coo, hid = pool.reinit_and_sample()

    nca.build_io_tensors(input_electrodes=in_coo,
                          output_electrodes=out_coo,
                          hidden_inputs=hid,
                          custom_input_signal=False,
                          output_alive=False)

    dist_mask = create_distance_mask(out_coo[0],N=nca.size, shape="circle")

    x, loss = train_step(nca,x,trainer,
                         distance_mask=dist_mask,
                         custom_channel=False,
                         custom_channel_val=None,
                         min_step=MIN_STEP,
                         max_step=MAX_STEP,
                         energy_budget=ENERGY_BUDGET)

    ar = x.numpy()
    if np.random.random() <0.2:
      #we replace the worst sample by the initial state
      batch_losses,task_loss, energy_cost = loss_distance_to_output(x, dist_mask,energy_budget=ENERGY_BUDGET,k=20, return_array=True)
      batch_losses = batch_losses.numpy()
      worst_ind = np.argmax(batch_losses)
      ar[worst_ind] = get_initial_state(None, nca.size, nca.channel_n)


    pool.commit(ar)

    nca.loss_history.append(loss.numpy())
    if i%10 == 0:
      batch_losses,task_loss, energy_cost = loss_distance_to_output(x, dist_mask,energy_budget=0,k=16, return_array=True)
      print(("\rStep #%d |Loss: %.2f |"+\
              "Energy at target: %.2f |"+ \
              "Proximity Score: %.2f |"+ \
              "Energy cost: %.2f      ")%(i,loss,
                        ar[1][nca.out_cells[0][0]][nca.out_cells[0][1]][1],
                        task_loss[0], energy_cost[0]), end='')

## Visualize the Model

#Uncomment to load a pretrained model
nca.dmodel.load_weights("PretrainedModel/ExploreEnvironmentModel")

# At low energy, you should observe gliders values exploring the whole grid and fixing
# on the output when they find it
# At high energy
x, out_coo, in_coo, hid = pool.reinit_and_sample()

nca.build_io_tensors(input_electrodes=nca.in_cells,
                      output_electrodes=[(5,25)],
                      hidden_inputs=hid,
                      custom_input_signal=False,
                      output_alive=False)

nca.visualize_ca(x, nb_step=400,
                      channels = [1,0],
                      titles={1:"Energy channel", 0:"Hidden channel"},
                      init_wait=5., output_video=False,
                      custom_bounds={0:(-0.7,0.7), 1:(-0.7,0.7)})


## We can also expand the grid

# we save the weights
nca.dmodel.save_weights("ExploreEnvironmentModel")

# we created a new model and a new pool with bigger grids

nca_big_grid = NeuralCA(use_per_cell_fire_rate=True, input_electrodes = [(64,64)],
                 output_electrodes = [(65,64)], output_visible=True,output_alive=False,
                 custom_input_signal=False,batch_size=8, size=128, channel_n=4,
                 fire_rate=0.9, first_lay_size= 10,nn_init="random", bound_values=False,
                 value_bounds=(-2,2), all_cells_alive=False, custom_channel_nb = 0,
                 enforce_connexity=False, energy_as_amplitude=False, max_amplitude=None,
                 bound_energy_change=False, torus_boundaries=False, add_noise=False,
                 use_hidden_inputs=False)

nca_big_grid.dmodel.load_weights("ExploreEnvironmentModel")


pool_big_grid = BatchPool(20, nca_big_grid.size, nca_big_grid.channel_n,
                nca_big_grid.in_cells, nca_big_grid.out_cells,  radius=50,
                batch_size=nca_big_grid.batch_size, range_nb_hidden_cells=(0,0),
                damage=False, proba_move=1., move_inputs=False,
                move_outputs=True, init_type="zeros")

#visualize again

x, out_coo, in_coo, hid = pool_big_grid.reinit_and_sample()

nca_big_grid.build_io_tensors(input_electrodes=nca_big_grid.in_cells,
                      output_electrodes=out_coo,
                      hidden_inputs=hid,
                      custom_input_signal=False,
                      output_alive=False)

nca_big_grid.visualize_ca(x, nb_step=400,
                      channels = [1,0],
                      titles={1:"Energy channel", 0:"Hidden channel"},
                      init_wait=5., custom_bounds={1:(-0.7,0.7), 0:(-0.7,0.7)},
                      output_video=False, recreate_fig_each_step=False)














