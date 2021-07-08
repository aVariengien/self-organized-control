import tensorflow as tf
import numpy as np
import time
from SelfOrgControl.NeuralCA import *



lr = 1e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [3000, 5000], [lr, lr*0.1, lr*0.001])


inp_cell_pos = [(13, 16),(19,16)]
out_cell_pos = [(16,19)]
nca = TrainableNeuralCA(input_electrodes = inp_cell_pos,
                            output_electrodes = out_cell_pos,
                            grid_size=32,
                            batch_size=16, channel_n=6,
                            ca_steps_per_sample=(50,60),
                            replace_proba=0.01,
                            task_loss_w=0.5, grid_pool_size=100,
                            learning_rate=lr,
                            repeat_input=1, #there is no redondancy
                            torus_boundaries=False,
                            penalize_overflow=True, overflow_w = 1e1,
                            use_hidden_inputs=True, perturb_io_pos=True,
                            add_noise=False, damage=False,
                            nb_hid_range=(0,0), move_rad=1, proba_move=1.0)
# We add perturbation in the cell positions

print(nca.neuralCA.dmodel.summary())


inputs_b = np.random.randint(0,2,(6000,16,2)) *2 -1
targets_b = np.expand_dims(np.prod(inputs_b, axis=-1), axis=-1)
#The product gives the not xor boolean function if we interpret -1->0, 1->1.
#You can invert the product to get the xor gate


nca.fit(inputs_b, targets_b, verbose=True, use_batch=True)

nca.neuralCA.dmodel.save_weights("XOR_Pertubed_IO_pos")

#check if the training worked
nca.plot_losses()
nca.plot_io_signals(55,inputs_b[:10,0,:], targets_b[:10,0,:], add_plot_jitter=True)

## Fine-tuned the previous model to be robust to more perturbation

lr = 1e-4
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [3000, 5000], [lr, lr*0.1, lr*0.001])

nca_more_perturb = TrainableNeuralCA(input_electrodes = inp_cell_pos,
                            output_electrodes = out_cell_pos,
                            grid_size=32,
                            batch_size=16, channel_n=6,
                            ca_steps_per_sample=(50,60),
                            replace_proba=0.01,
                            task_loss_w=0.5, grid_pool_size=100,
                            learning_rate=lr,
                            repeat_input=1, #there is no redondancy
                            torus_boundaries=False,
                            penalize_overflow=True, overflow_w = 1e1,
                            use_hidden_inputs=True, perturb_io_pos=True,
                            add_noise=True, damage=True,
                            nb_hid_range=(0,0), move_rad=2, proba_move=1.0)


nca_more_perturb.neuralCA.dmodel.load_weights("NOTXOR_Pertubed_IO_pos")

nca_more_perturb.fit(inputs_b, targets_b, verbose=True, use_batch=True)



## Visualisation

#Load pretrained model and visualize it
nca_more_perturb.neuralCA.dmodel.load_weights("PretrainedModel/XOR_Pertubed_IO_pos_damage_noise")

nca_more_perturb.plot_io_signals(55,inputs_b[:10,0,:], targets_b[:10,0,:],
                                                        add_plot_jitter=True)

nca_more_perturb.visualize_nca_predict(nb_step_per_input=50, inputs=inputs_b[5:10,0,:],
                                        channels=[0,1,4,5],
                                        custom_bounds={0:(-1.1, 1.1)},
                                        titles={0:"Information Channel",
                                                  1:"Hidden channel #1",
                                                  4:"Hidden channel #2",
                                                  5:"Hidden channel #3"},
                                        output_video=False,
                                        recreate_fig_each_step=False)



