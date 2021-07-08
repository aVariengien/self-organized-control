import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
import matplotlib.pyplot as plt
import json
import time


ALL_TIMES = {}
def init_chrono(t_id):
    global ALL_TIMES
    ALL_TIMES[t_id] = time.process_time()

def get_chrono(t_id):
    global ALL_TIMES
    t = time.process_time()
    print("-- "+t_id+" - chrono: ", ALL_TIMES[t_id]-t)
    ALL_TIMES[t_id] = 0

def save_to_video(name,image_folder,fps=4, form="png"):
    os.system(r"ffmpeg -r "+str(fps)+" -i "+image_folder+r"/img_%01d."+\
            form+" -vcodec mpeg4 -y -qscale 0 "+str(name)+".mp4")

def get_random_id():
    r = str(np.random.random())
    return r[-5:]


def get_rolling_avg(a, k=10):
    s= np.sum(a[:k])
    rol_avg  = [s/k]
    for i in range(k,len(a)):
        s += a[i]
        s -= a[i-k]
        rol_avg.append(s/k)
    return np.array(rol_avg)


@tf.function
def get_living_mask(x):
    energy = x[:, :, :, 1:2]
    return tf.nn.max_pool2d(energy, 3, [1, 1, 1, 1], 'SAME') > 0.1

@tf.function
def double_ramp(x,A=0.05,B=0.1,C=1):
    x = tf.clip_by_value(x, -2, 2)
    pos = tf.cast(x>0, tf.float32)
    neg = 1-pos
    return (0.5*(C-B)*x + B)*pos + (0.5*(B-A)*x + B)*neg


@tf.function
def get_connected_cells(cell_pos, input_pos, size):
    connected_comp = input_pos
    for step in tf.range(0, size):
        connected_comp = tf.nn.max_pool2d(connected_comp,3, [1, 1, 1, 1], "SAME")*cell_pos
    connected_comp = tf.nn.max_pool2d(connected_comp,3, [1, 1, 1, 1], "SAME")
    #we add a last maxpool in order to include the neighboor of the living
    #cells connected to the input
    return connected_comp

@tf.function
def ca_loss(grids, ca, step_target, task_w=100,
            count_positive_nrj=True, overflow_w=1e3,
            overflow_cost = True):

    out_cell_state = ca.get_output_cell_states(grids)

    if overflow_cost: #we penalize channel values out of [-5,5] to stabilize the dynamics
        overflow = tf.reduce_mean(tf.square(tf.clip_by_value(grids, -5., 5.)-grids),  axis=[1,2,3])

    task_loss = tf.math.reduce_mean(tf.square(out_cell_state[:,:,0]-step_target), axis=[1])

    if overflow_cost:
        batch_losses = task_w*task_loss + overflow_w * overflow
    else:
        batch_losses = task_w*task_loss
    return batch_losses

def get_initial_state(N, size, chan, init_type="zeros"):
    if init_type == "random":
        if N is None:
            return (np.random.random([size, size, chan])-0.5)*0.2
        x= (np.random.random([N,size, size, chan])-0.5)*0.2
    elif init_type == "zeros":
        if N is None:
            return np.zeros([size, size, chan])
        x= np.zeros([N,size, size, chan])

    return x.astype("float32")


def discrete_2D_rand_vect(center, R):
    angle=np.random.random()*2*np.pi
    radius = R * np.sqrt(np.random.random())
    unit_vect = np.array([np.cos(angle), np.sin(angle)])
    v=np.round(unit_vect*radius + center)
    v = list(v)
    v[0] = int(v[0])
    v[1] = int(v[1])
    return v

def perturb_position(positions,radius=2, proba_move=1.):
    perturb_pos = []
    for i,p in enumerate(positions):
        if np.random.random() < proba_move:
            perturb_pos.append(discrete_2D_rand_vect(p,radius))
        else:
            perturb_pos.append(p)
    return perturb_pos

def random_choice(l, nb_sample):
    if len(l) == 0:
        return []
    samples_idx=np.random.choice(np.arange(len(l)), nb_sample, replace=False)
    samples = []
    for x in list(samples_idx):
        samples.append(l[x])
    return samples

import tensorflow as tf

@tf.function
def damage_grids(grids):
    n = grids.shape[0]
    h = grids.shape[1]
    w = grids.shape[2]
    c = grids.shape[3]
    x = tf.linspace(-1.0, 1.0, w)[None, None, :]
    y = tf.linspace(-1.0, 1.0, h)[None, :, None]
    center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
    r = tf.random.uniform([n, 1, 1], 0.3, 0.4)
    x, y = (x-center[0])/r, (y-center[1])/r
    mask = tf.cast(x*x+y*y > 1.0, tf.float32)
    mask = tf.expand_dims(mask,-1)
    dam_grids = mask*grids + (1.-mask)*tf.random.uniform((n,h,w, c), -1., 1.)
    return dam_grids




class BatchPool:
    """A pool of batches. With each batch comes the output cells coordinates
        attached and the list of hidden cells."""
    def __init__(self, pool_size, grid_size,nb_chan,output_centers,
                 input_centers, radius, batch_size,
                 range_nb_hidden_cells=(0,3), damage=False,
                 proba_move=1., move_inputs=True,
                 move_outputs=True, init_type="random"):
        """pool_size: the number of batches in the pool"""
        self.grid_size = grid_size
        self.nb_chan = nb_chan
        self.range_nb_hidden_cells = range_nb_hidden_cells
        self.proba_move_out = proba_move*move_outputs
        self.proba_move_in = proba_move*move_inputs

        self.init_type = init_type

        self.size = pool_size
        self.damage = damage
        self.radius = radius
        self.output_centers = output_centers
        self.input_centers = input_centers
        self.batch_size = batch_size
        self.current_idx = None
        self.batches = get_initial_state(pool_size*batch_size, grid_size,nb_chan, init_type=self.init_type)
        self.batches = self.batches.reshape(pool_size,batch_size, grid_size, grid_size, nb_chan)
        self.batches_output_coo = [perturb_position(output_centers,radius,self.proba_move_out) for i in range(pool_size)]
        self.batches_input_coo = [perturb_position(input_centers,radius,self.proba_move_in) for i in range(pool_size)]

        self.batches_hidden_inputs = []
        for i in range(len(self.batches_input_coo)):
            nb_hid = np.random.randint(range_nb_hidden_cells[0], range_nb_hidden_cells[1]+1)
            hid = random_choice(self.batches_input_coo[i], nb_hid)

            self.batches_hidden_inputs.append(hid)

    def sample(self):
        idx = np.random.randint(self.size)
        self.current_idx = idx #the idicies of the samples currently under update

        if self.damage and np.random.random() < 0.5:
            grids = damage_grids(self.batches[idx])
        else:
            grids = self.batches[idx]

        return (grids, self.batches_output_coo[idx],
               self.batches_input_coo[idx], self.batches_hidden_inputs[idx])
    def reinit_and_sample(self):
        idx = np.random.randint(self.size)
        self.current_idx = idx
        self.batches_output_coo[idx] = perturb_position(self.output_centers,
                                                self.radius,self.proba_move_out)
        self.batches_input_coo[idx] = perturb_position(self.input_centers,
                                                self.radius,self.proba_move_in)

        nb_hid = np.random.randint(self.range_nb_hidden_cells[0],
                                  self.range_nb_hidden_cells[1]+1)
        hid = random_choice(self.batches_input_coo[idx], nb_hid)

        self.batches_hidden_inputs[idx] = hid
        self.batches[idx] = get_initial_state(self.batch_size, self.grid_size,
                                              self.nb_chan, init_type=self.init_type)
        return (self.batches[idx], self.batches_output_coo[idx],
               self.batches_input_coo[idx], self.batches_hidden_inputs[idx])

    def commit(self, updated_batch):
        self.batches[self.current_idx] = updated_batch
        self.current_idx = None




class TrainableNeuralCA():
    """A neural CA that can be trained like a usual machine learning model."""


    def __init__(self, input_electrodes,
                    output_electrodes,grid_size=32, batch_size=16,
                  channel_n=6, ca_steps_per_sample=(20,50),
                  replace_proba=0.5,
                  task_loss_w=100, grid_pool_size=100,
                  learning_rate=1e-2,
                  torus_boundaries=False, repeat_input=1,
                  penalize_overflow=False, overflow_w = 1000.0,
                  use_hidden_inputs =False, nb_hid_range=(0, 0),
                  perturb_io_pos=False,move_rad=0, proba_move=1.,
                  add_noise=False, damage =False):

        """
            input_electrodes -- The list of the input cells positions.
                                The order determine the matching between input values
                                and the corresponding input cell. If there is x
                                amount of redundancy, then the x first elements
                                correspond to the first observation, the x
                                that follow to the second etc.
            output_electrodes -- The list of the output cells positions. the order
                                determine the order in the outputs vectors returned.
            grid_size -- the size of the square CA grid
            channel_n -- the number of the channels in the cell state
            ca_steps_per_sample -- the range in wich uniform sampled
                                    the number of iteration to run the
                                    CA between input update and readout
            replace_proba -- The probability of replacing the worst element of a batch
                            with a random grid
            task_loss_w -- The weight to ponderate the L2 loss of the task
            grid_pool_size -- the size of the pool of grids
            learning_rate -- learning rate for the Adam optimizer
            torus_boundaries -- whether to use cyclic boundaries
            repeat_input -- The amount of redondancy in the input. If equal to x,
                            each observation will be mapped to x input cells.
            penalize_overflow -- whether to add penality to channels values out
                                    of bounds [-5,5]
            overflow_w -- The weight to ponderate the overflow penality
            use_hidden_inputs -- whether to hid some input cells
            nb_hid_range -- The range in wich to chose the number of hidden input
                            cells for each batch in the pool.
            perturb_io_pos -- Whether to add random perturbation to the position
                                of input and output cells.
            move_rad -- The radius of the perturbation.
            proba_move -- The probability that a input or output cell will be displaced.

            add_noise -- Whether to use noisy update
            damage -- Whther to damage part (half) of the grids in the pool
        """

        self.neuralCA = NeuralCA(input_electrodes=input_electrodes,
                                output_electrodes = output_electrodes,
                                custom_input_signal = True, batch_size =batch_size,
                                size = grid_size, channel_n = channel_n,
                                bound_values = False,value_bounds=None,
                                all_cells_alive = True, torus_boundaries=torus_boundaries,
                                use_hidden_inputs = use_hidden_inputs, add_noise=add_noise)

        self.perturb_io_pos=perturb_io_pos
        self.overflow_w = overflow_w
        self.penalize_overflow = penalize_overflow
        self.replace_proba = replace_proba
        self.grid_size = grid_size
        self.ca_nb_channel = channel_n
        self.task_loss_w = task_loss_w
        self.ca_steps_per_sample = ca_steps_per_sample
        self.batch_size = batch_size


        self.grid_pool = BatchPool(grid_pool_size, grid_size, channel_n,
                                    output_electrodes,input_electrodes,
                                    move_rad, batch_size,
                                    range_nb_hidden_cells=nb_hid_range,
                                    damage=damage, proba_move=proba_move)


        self.grid_pool_size = grid_pool_size
        self.repeat_input = repeat_input
        self.loss_history = []

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.nb_sample_seen = 0

    def load(self,filename, no_json=False):
        self.neuralCA.load_weights(filename)
        if not(no_json):
            with open(filename+".json", "r") as f:
                d = json.loads(f.read())
            for x in d:
                setattr(self, x, d[x])
        self.neuralCA.loss_history = self.loss_history.copy()

    def save(self,filename):
        """save the weights using tensorflow paramters save
            and other hyperparameters about the model in a json file"""
        self.neuralCA.save_weights(filename)
        self.loss_history = [float(f) for f in list(self.neuralCA.loss_history)]
        d = self.__dict__.copy()
    #to avoid non tring convertable object
        del(d['neuralCA'])
        del(d['optimizer'])
        del(d['grid_pool'])


        try:
            with open(filename+".json", "w") as f:
                f.write(json.dumps(d))
        except TypeError:
            print(d)
            print("Error for json conversion")


    def fit(self, inputs, teachers, verbose=False, use_batch=False):
        """Perform gradient deseint step given the teachers vector and the input provided.

        use_batch -- if False, each input will be match to batch_size different grids.,
                     if True, each input is matched with one grid, organized in batches
                     the length of the input must be a multiple of the batch size.
                     In all cases, we use batches for the CA grids."""

        t_init = time.process_time()
        if verbose:
            print("Training ...")
        avg_loss = 0
        t1 = time.process_time()
        for i in range(len(inputs)):
            self.nb_sample_seen +=1
            step_loss = self.train_one_sample(inputs[i], teachers[i], use_batch)

            if verbose and i%10 == 0:
                t2 = time.process_time()
                print("\r Training: %.2f %% | Time per sample: %.3fs | log10 of loss: %.3f    "%(100*i/len(inputs),(t2-t1)/10, np.log10(step_loss)),end='')
                t1 = time.process_time()

            self.neuralCA.loss_history.append(step_loss.numpy())
            avg_loss += step_loss.numpy()


        t_end = time.process_time()
        if verbose:
            print("Training end in %.3fs. Mean log 10 loss: %.3f"%(t_end-t_init, np.log10(avg_loss/len(inputs))))
        return avg_loss/len(inputs)

    def change_to_ca_shape(self, vect, batch=False, is_input=False):
        """Change a 1D tensor of shape (LEN) by repeating its value to get the final
        shape (batch_size, LEN)"""
        ca_vect = vect
        if not(batch):
            ca_vect  = np.expand_dims(ca_vect, 0)
            ca_vect = ca_vect.repeat(self.batch_size, axis=0)
        if is_input:
            ca_vect = ca_vect.repeat(self.repeat_input, axis=-1)
        return ca_vect.astype("float32")


    def train_one_sample(self,input_val, teacher, use_batch=False):
        if not(use_batch):
            assert input_val.shape[0]*self.repeat_input == len(self.neuralCA.in_cells)
            assert teacher.shape[0] == len(self.neuralCA.out_cells)
            ca_in = self.change_to_ca_shape(input_val, is_input=True)
            ca_targ = self.change_to_ca_shape(teacher)
        else:
            assert input_val.shape[0] == self.batch_size
            assert teacher.shape[0] == self.batch_size
            assert input_val.shape[1]*self.repeat_input == len(self.neuralCA.in_cells)
            assert teacher.shape[1] == len(self.neuralCA.out_cells)
            ca_in = self.change_to_ca_shape(input_val, batch=True, is_input=True)
            ca_targ = self.change_to_ca_shape(teacher, batch=True)

        if self.perturb_io_pos:
            if np.random.random() < 0.1:
                grids, out_pos, in_pos, hid = self.grid_pool.reinit_and_sample()
            else:
                grids, out_pos, in_pos, hid = self.grid_pool.sample()
            self.neuralCA.build_io_tensors(input_electrodes=in_pos, output_electrodes=out_pos,
                                            hidden_inputs=hid, custom_input_signal=True)

        else:
            grids = self.grid_pool.sample(self.batch_size)

        final_grids, loss = self.ca_train_step(ca_in, ca_targ, grids)
        final_grids_np = final_grids.numpy()

        if np.random.random() < self.replace_proba:
            ca_batch_losses = ca_loss(final_grids,self.neuralCA,ca_targ,
                                      task_w=self.task_loss_w,
                                       overflow_w=self.overflow_w,
                                      overflow_cost = self.penalize_overflow)

            ca_batch_losses = ca_batch_losses.numpy()
            worst_ind = np.argmax(ca_batch_losses)
            final_grids_np[worst_ind] = get_initial_state(None, self.grid_size, self.ca_nb_channel) #we replace the worst sample by the initial state

        self.grid_pool.commit(final_grids_np)
        return loss


    @tf.function
    def run_ca(self,input_val, grids, nb_ca_steps):
        input_val = tf.cast(input_val, tf.float32)
        for i in tf.range(nb_ca_steps):
            grids = self.neuralCA(grids, updates = input_val)
        return grids

    def run_ca_step_by_step(self,input_val, grids, nb_ca_steps,
                          conti_damage_proba=None, keep_all_grids=False):
        input_val = tf.cast(input_val, tf.float32)
        all_grids = [grids]

        for i in range(nb_ca_steps):

            grids = self.neuralCA(grids, updates = input_val)
            if conti_damage_proba is not None:
                if np.random.random() < conti_damage_proba:
                    grids = damage_grids(grids)

            if keep_all_grids:
                all_grids.append(grids)

        return grids, all_grids

    @tf.function
    def ca_train_step(self,input_val, teacher, grids):
        nb_ca_steps = tf.random.uniform([], self.ca_steps_per_sample[0],
                                  self.ca_steps_per_sample[1], tf.int32)

        with tf.GradientTape() as g:
            grids = self.run_ca(input_val, grids, nb_ca_steps)
            losses = ca_loss(grids,self.neuralCA,teacher,
                              task_w=self.task_loss_w,
                              overflow_w=self.overflow_w,
                              overflow_cost = self.penalize_overflow)
            loss = tf.math.reduce_sum(losses)

        grads = g.gradient(loss, self.neuralCA.dmodel.weights)
        grads = [g/(tf.norm(g)+1e-8) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.neuralCA.weights))

        return grids, loss


    def predict(self,Xs, nb_ca_steps=None, sample_from_pool=True,
                use_batch=False, return_all_grids=False, init_grids=None,
                no_reinit=False, conti_damage_proba=None,
                return_final_grids=False):
        """

        Xs -- The inputs of the neural CA of shape (nb_inputs, input_shape)

        nb_ca_steps -- The number of steps to run the ca before readout

        sample_from_pool -- are the initial grids sampled from pool or initialized

        use_batch -- If use batch, the len of Xs must be a multiple of the ca batch size.
                     Each input will be match to one grid that will then be arranged
                    into batches.

        return_all_grids -- Return a list of the intermediate grids steps, useful
                            for visualisations

        init_grids -- If not None, these grids are used as the initial step
                      for the CA. It must have shape
                      (batch_size, ca_size, ca_size, nb_channels).

        no_reinit -- If True, no grid will be reinitialized during the simulations.

        conti_damage_proba -- If not none, the number define a probability of
                              damaging the grid at each steps

        return_final_grids -- If true, the final state of the grids after each
                                prediction are returned, organized in batches
                                if use_batch is true.

        """
        if nb_ca_steps is None:
            nb_ca_steps = np.random.randint(self.ca_steps_per_sample[0],
                                            self.ca_steps_per_sample[1])
            #if no nb of steps is defined, we take the higher bound of the training
        nb_ca_steps = tf.convert_to_tensor(nb_ca_steps)

        outputs = []
        output_grids= []

        if use_batch:
            step_size = self.batch_size
        else:
            step_size = 1

        for i in range(0,len(Xs), step_size):
            if use_batch:
                ca_input = self.change_to_ca_shape(np.array(Xs[i:i+step_size]),
                                                  is_input=True, batch=True)
            else:
                ca_input = self.change_to_ca_shape(Xs[i], is_input=True)

            if init_grids is not None: #we can specify the starting state of the grids
                grids = init_grids
            elif sample_from_pool:
                if self.perturb_io_pos:
                    if np.random.random() < self.replace_proba and not(no_reinit):
                        grids, out_pos, in_pos, hid = self.grid_pool.reinit_and_sample()
                    else:
                        grids, out_pos, in_pos, hid = self.grid_pool.sample()
                    self.neuralCA.build_io_tensors(input_electrodes=in_pos, output_electrodes=out_pos,
                                                  hidden_inputs=hid, custom_input_signal=True)
                else:
                    grids = self.grid_pool.sample(self.batch_size)

            else:
                grids = get_initial_state(self.batch_size, self.grid_size,
                                        self.ca_nb_channel)

            #we use a function that runs the CA without the @tf.function
            #optimizing the computing speed to be able to modify / keep
            #every intermediae grid state
            if return_all_grids or conti_damage_proba is not None:
                final_grids, all_grids = self.run_ca_step_by_step(ca_input,
                                                                   grids,
                                                                   nb_ca_steps,
                                                                conti_damage_proba,
                                                                keep_all_grids=return_all_grids)
            else: #quick CA executing
                final_grids = self.run_ca(ca_input, grids, nb_ca_steps)


            self.grid_pool.commit(final_grids)

            if return_all_grids:
                output_grids.append(all_grids)
            elif return_final_grids:
                output_grids.append(final_grids)

            outputs.append(self.neuralCA.get_output_cell_states(final_grids))

        if use_batch:
            outputs = np.array(outputs)
            shape_o = outputs.shape
            outputs = outputs.reshape((len(Xs),shape_o[2], shape_o[3]))
            outputs = outputs[:,:,0]

            if return_final_grids:
                output_grids = np.array(output_grids)
                shape_g = output_grids.shape
                output_grids = output_grids.reshape((len(Xs),nb_ca_steps, shape_g[2], shape_g[3], shape_g[4]))

        else:
            outputs =  np.array(outputs)[:,0,:,0]
            if return_final_grids:
                output_grids = np.array(output_grids)

        if return_final_grids or return_all_grids:
            return outputs, output_grids
        else:
            return outputs

    def plot_losses(self, beg=0, avg_on=20, high_bound=None):
        """Alias for the NeuralCA.plot_losses method"""
        self.neuralCA.plot_losses(beg=beg, avg_on=avg_on, high_bound=high_bound)



    def plot_io_signals(self, nb_step_per_input, inputs,targets=None,
                    plot_log=False, add_plot_jitter=False, frames=None,
                    use_precomputed_frames=False, plot_input=True):
        """Create plot of input/output activation of the neural CA."""
        if add_plot_jitter:
            l = 0.95
            h = 1.05
        else:
            l=1
            h=1
        nb_step = nb_step_per_input*len(inputs)
        input_seq = np.repeat(inputs, nb_step_per_input, axis=0)
        input_seq = self.change_to_ca_shape(input_seq, is_input=True)


        if targets is not None:
            target_seq = np.repeat(targets, nb_step_per_input, axis=0)
            target_seq = self.change_to_ca_shape(target_seq)

        x = get_initial_state(self.neuralCA.batch_size, size=self.neuralCA.size, chan=self.neuralCA.channel_n)

        frames= []
        step_losses = []
        inputs = [[] for i in range(len(self.neuralCA.in_cells))]
        outputs = []

        for i in range(nb_step):
            if not(use_precomputed_frames):
                x = self.neuralCA(x, updates=input_seq[:,i])
            else:
                x = frames[i]
            x_np = x.numpy()
            x_np = x.numpy()
            frames.append(x_np[0,:,:,0])

            #get loss stats
            if targets is not None:
                losses= ca_loss(x,self.neuralCA, step_target=target_seq[:,i],
                                task_w=self.task_loss_w,
                                overflow_w=self.overflow_w,
                                overflow_cost = self.penalize_overflow)


                step_losses.append(losses[0].numpy())

            for kk in range(len(self.neuralCA.in_cells)):
                inputs[kk].append(x_np[0,self.neuralCA.in_cells[kk][0], self.neuralCA.in_cells[kk][1], 0])
            outputs.append(self.neuralCA.get_output_cell_states(x)[0,:,0])

        step_losses = np.array(step_losses)
        if plot_log:
            step_losses = np.log10(step_losses)

        if targets is not None:
            fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(15,6))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15,6))
            axes = [axes]

        if targets is not None:
            axes[1].plot(step_losses, label= "log of "*plot_log+"Loss")
            axes[1].set_xlabel("CA step")
            axes[1].set_title("Evolution of the loss during the run")
            axes[1].legend()

        inputs = np.array(inputs)
        outputs = np.array(outputs)
        if plot_input:
            for kk in range(len(self.neuralCA.in_cells)):
                axes[0].plot(inputs[kk]*np.random.uniform(l,h),alpha=0.6, label= "Input "+str(kk))

        for kk in range(len(self.neuralCA.out_cells)):
            if targets is not None:
                axes[0].plot(target_seq[0,:nb_step,kk]*np.random.uniform(l,h), label= "Target signal "+str(kk))
            axes[0].plot(outputs[:,kk], label= "Output signal "+str(kk))
        axes[0].legend()
        axes[0].set_xlabel("CA step")
        axes[0].set_title("Input and output signals")
        fig.tight_layout()
        fig.show()
        return frames

    def visualize_nca_predict(self, nb_step_per_input=30, inputs=None,
                    frames=None, dt=0.03,
                    init_wait=0.01, channels=[0,1],
                    custom_bounds={}, verbose=False,
                    recreate_fig_each_step=False,
                    init_state_from_pool=False,titles={},
                    output_video=False, img_folder="video_images/",
                    video_name="CA_video", fps=6, dpi=100, form="png",
                    img_step=1):
        """Prepare the inputs to be passed to the method NeuralCA.visualize_ca"""
        input_seq = np.repeat(inputs, nb_step_per_input, axis=0)
        input_seq = self.change_to_ca_shape(input_seq, is_input=True)

        nb_step = nb_step_per_input * len(inputs)

        if init_state_from_pool:
            x, out_pos, in_pos, hid = self.grid_pool.sample()
            self.neuralCA.build_io_tensors(input_electrodes=in_pos, output_electrodes=out_pos,
                                            hidden_inputs=hid, custom_input_signal=True)
        else:
            x, out_pos, in_pos, hid = self.grid_pool.reinit_and_sample()
            self.neuralCA.build_io_tensors(input_electrodes=in_pos, output_electrodes=out_pos,
                                            hidden_inputs=hid, custom_input_signal=True)

        self.neuralCA.visualize_ca(init_state=x,
                    nb_step=nb_step,
                    input_seq=input_seq,
                    frames=frames, dt=dt,
                    init_wait=init_wait, channels=channels,
                    custom_channel_val=None,titles=titles,
                    custom_bounds=custom_bounds, verbose=verbose,
                    recreate_fig_each_step=recreate_fig_each_step,
                    output_video=output_video, img_folder=img_folder,
                    video_name=video_name, fps=fps, dpi=dpi, form=form,
                    img_step=img_step)



@tf.function
def cyclic_padding(x, pad):
    x = tf.concat([x[:, -pad:], x, x[:, :pad]], 1)
    x = tf.concat([x[:,:, -pad:], x, x[:, :, :pad]], 2)
    return x

class NeuralCA(tf.keras.Model):
    def __init__(self, use_per_cell_fire_rate=False, input_electrodes = [(0,0)],
                 output_electrodes = [(1,0)], output_visible=True,output_alive=False,
                 custom_input_signal=False,batch_size=8, size=32, channel_n=6,
                 fire_rate=0.5, first_lay_size= 30, nn_init="zeros", bound_values=False,
                 value_bounds=None, all_cells_alive=False, custom_channel_nb = 0,
                 enforce_connexity=False, energy_as_amplitude=False, max_amplitude=None,
                 bound_energy_change=False, torus_boundaries=False, add_noise=False,
                 use_hidden_inputs=False):
        """
            use_per_cell_fire_rate -- If true the channel 1 will be interpreted
                                    as a per-cell firerate
            output_visible -- If output identifier channel (channel 3) should
                                be used
            output_alive -- only relevant for cases where the channel 1 is
                                describing "energy".
            custom_input_signal -- Whether to input values in the information
                                    channel of input cells. If false, the inform
                                    ation channel will be constant equals to 1.
            fire_rate -- The fire rate of the CA: the probability that
                        a cell will update at a given step. It is added to the
                        per-cell fire rate if use_per_cell_fire_rate is True.
            first_lay_size -- the size of the first layer of the neural network
                                controlling the update
            nn_init -- The initialisation of the update neural net. "zeros":
                        nothing happends by default or "random".
            bound_values -- whether to clip values that are out of bounds
            value_bounds -- the bound in wich to keep values if bound_values is True
            all_cells_alive -- If false, the living cell are determined as described
                                in the orginal NCA paper: only neighbors of
                                living cell can become alive. Inputs are always
                                alive.
            custom_channel_nb -- The channel in wich to hardcode an array such
                                as a gradient.
            enforce_connexity -- If true, all the cell that are not connected
                                by a path (diagonal, vertical, horizontal moves
                                are allowed) of alive cells to an input cell are deleted
                                at each step.
            energy_as_amplitude -- If true, interprete the energy channel (channel 1)
                                    as an amplitude of update.
            max_amplitude -- If not None, the maximal absolute value for an update.
            bound_energy_change -- If true, the maximal amplitude will also affect the
                                    energy channel.

        """

        super().__init__()
        self.energy_as_amplitude = energy_as_amplitude
        self.enforce_connexity = enforce_connexity
        self.bound_values = bound_values
        self.value_bounds = value_bounds
        self.max_amplitude = max_amplitude
        self.bound_energy_change = bound_energy_change
        self.add_noise = add_noise

        self.use_hidden_inputs = use_hidden_inputs

        self.torus_boundaries = torus_boundaries


        self.all_cells_alive = all_cells_alive
        self.use_per_cell_fire_rate = use_per_cell_fire_rate
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.size = size
        self.batch_size = batch_size
        self.shape = tf.constant([batch_size,size, size, channel_n])
        self.first_lay_size = first_lay_size
        self.loss_history = [] #list of the log of the loss of the NCA during training

        #dummy initialization of the tf.Variable to fix the shapes
        self.in_coo_tensor = tf.Variable(tf.zeros((batch_size*len(input_electrodes), 4),tf.int32), trainable=False)
        self.out_coo_tensor = tf.Variable(tf.zeros((batch_size*len(output_electrodes), 3), tf.int32), trainable=False)
        self.cut_io_mask = tf.Variable(tf.ones((batch_size,self.size,self.size, self.channel_n), tf.float32), trainable=False)
        self.pos_io_channel = tf.Variable(tf.ones((batch_size,self.size,self.size, self.channel_n), tf.float32), trainable=False)

        self.hidden_inputs_mask = tf.Variable(tf.ones((batch_size,self.size,self.size, self.channel_n), tf.float32), trainable=False)

        self.build_io_tensors(input_electrodes,output_electrodes,custom_input_signal, output_visible,output_alive)

        if self.torus_boundaries:
            padding_type = 'VALID'
        else:
            padding_type = 'SAME'

        if nn_init == "zeros":
            initializer_nn = tf.zeros_initializer
        elif nn_init == "random":
            initializer_nn = tf.initializers.GlorotUniform

        self.dmodel = tf.keras.Sequential([
              #DepthwiseConv2D(kernel_size = (3,3), padding=padding_type, depth_multiplier=6,activation=tf.nn.relu, use_bias=True), #we add here the trainable perception filter
              Conv2D(kernel_size = (3,3), filters=20, padding='SAME',activation=tf.nn.relu),
              Conv2D(self.first_lay_size, 1, activation=tf.nn.relu),
              Conv2D(self.channel_n, 1, activation=None, kernel_initializer=initializer_nn) #we remove kernel_initializer=tf.zeros_initializer, bias_initializer=tf.zeros_initializer
        ])



        #if we want to give custom value to channel "custom_channel_nb"
        self.custom_channel_nb = custom_channel_nb
        chan_mask = np.zeros((self.batch_size, self.size, self.size, self.channel_n), "float32")
        chan_mask[:,:,:,custom_channel_nb] = np.ones((self.batch_size, self.size, self.size))
        chan_mask_neg = 1-chan_mask
        chan_mask = tf.convert_to_tensor(chan_mask)
        chan_mask_neg = tf.convert_to_tensor(chan_mask_neg)

        self.chan_mask = tf.Variable(chan_mask, trainable=False)
        self.chan_mask_neg = tf.Variable(chan_mask_neg, trainable=False)


        energy_mask = np.zeros((self.batch_size, self.size, self.size, self.channel_n), "float32")
        energy_mask[:,:,:,1] = np.ones((self.batch_size, self.size, self.size))
        energy_mask_neg = 1-energy_mask
        energy_mask = tf.convert_to_tensor(energy_mask)
        energy_mask_neg = tf.convert_to_tensor(energy_mask_neg)

        self.energy_mask = tf.Variable(energy_mask, trainable=False)
        self.energy_mask_neg = tf.Variable(energy_mask_neg, trainable=False)



        self(tf.zeros([batch_size, size, size, channel_n]), updates = tf.zeros([batch_size*self.nb_in_cells]) )  # dummy call to build the model

    #TODO : Support the change in the number of input/ output when rebuilding the tensor
    def build_io_tensors(self,input_electrodes,output_electrodes,custom_input_signal,
                         output_visible=True, output_alive=False, hidden_inputs=None):
        """
            This function is used to define several tf.Variable that will be useful
            during the usage of the NeuralCA object. They include masks
            that implement the required behavior of input and output cells using
            tensor multiplications.
        """

        hidden_inputs_mask = np.ones(self.shape, "float32")
        if hidden_inputs is not None:
            if len(hidden_inputs) >0:
                for x,y in hidden_inputs:
                    hidden_inputs_mask[:,x,y,:] = np.zeros((self.batch_size,self.channel_n), "float32")

            self.hidden_inputs_mask.assign(tf.convert_to_tensor(hidden_inputs_mask))
        else:
            self.hidden_inputs_mask.assign(tf.ones(self.shape, tf.float32))


        self.custom_in_signal = custom_input_signal
        self.nb_in_cells = len(input_electrodes)
        self.nb_out_cells = len(output_electrodes)
        self.in_cells = input_electrodes #the x,y coordinate of the input cells
        self.out_cells = output_electrodes
        self.output_visible = output_visible
        self.output_alive = output_alive

        #the tensor of indices for the update of the input cells
        in_coo_tensor = []
        for i in range(self.batch_size):
            for x,y in self.in_cells:
                in_coo_tensor.append([i,x,y,0])
        in_coo_tensor = np.array(in_coo_tensor)
        in_coo_tensor = tf.cast(tf.convert_to_tensor(in_coo_tensor),tf.int32)
        self.in_coo_tensor.assign(in_coo_tensor)

        #the tensor of indices to gather the output states
        out_coo_tensor = []
        for i in range(self.batch_size):
            for x,y in self.out_cells:
                out_coo_tensor.append([i,x,y])
        out_coo_tensor = np.array(out_coo_tensor)
        out_coo_tensor = tf.cast(tf.convert_to_tensor(out_coo_tensor),tf.int32)
        self.out_coo_tensor.assign(tf.convert_to_tensor(out_coo_tensor))

        #the mask to remove in/out channels from all cells
        mask_cut_io_channels = np.ones((self.batch_size,self.size,self.size, self.channel_n), "float32")
        mask_cut_io_channels[:,:,:,2:4] = np.zeros((self.batch_size,self.size,self.size, 2), "float32")
        mask_cut_io_channels = tf.convert_to_tensor(mask_cut_io_channels)

        #the mask to remove the value of the input cells
        mask_cut_in_cells = np.ones((self.batch_size, self.size,self.size, self.channel_n), "float32")

        for (x,y) in self.in_cells:
            if hidden_inputs is not None:
                if [x,y] in hidden_inputs or (x,y) in hidden_inputs:

                    continue
            mask_cut_in_cells[:,x,y,:] = np.zeros((self.batch_size,self.channel_n), "float32")

        mask_cut_in_cells = tf.convert_to_tensor(mask_cut_in_cells)

        #we combine the two to limit the number of mul. at each call
        self.cut_io_mask.assign(mask_cut_in_cells * mask_cut_io_channels)

        #the mask to add the identification of the in/out cells + the value of the in cells if they are constant
        pos_io_channel = np.zeros((self.batch_size, self.size,self.size, self.channel_n), "float32")
        for (x,y) in self.in_cells:
            if hidden_inputs is not None:
                if [x,y] in hidden_inputs or (x,y) in hidden_inputs:
                    continue
            pos_io_channel[:,x,y,2] = np.ones(self.batch_size, "float32")   # 2 : the 'input cell' channel
            pos_io_channel[:,x,y,4:] = np.ones((self.batch_size, self.channel_n-4), "float32") #we set the hidden channel to 1.
            pos_io_channel[:,x,y,1] = np.ones(self.batch_size, "float32")   #an input cell is always alive
            if not(custom_input_signal):
                pos_io_channel[:,x,y,0] = np.ones(self.batch_size, "float32") #if we don't specify custom input signal,
                                                                              #the input is constant equals to 1
        if output_visible:
            for (x,y) in self.out_cells:
                pos_io_channel[:,x,y,3] = np.ones(self.batch_size, "float32")   # 3 : the 'output cell' channel
                pos_io_channel[:,x,y,4:] = np.ones((self.batch_size, self.channel_n-4), "float32") #we set the hidden channel to 1.
                if self.output_alive :
                    pos_io_channel[:,x,y,1] = np.ones(self.batch_size, "float32")   #If specified, an output can be alive

        pos_io_channel = pos_io_channel*hidden_inputs_mask #we remove the 1s identifying the inputs for the hidden
                                                                #inputs
        pos_io_channel = tf.convert_to_tensor(pos_io_channel)

        self.pos_io_channel.assign(pos_io_channel)

        return self.in_coo_tensor, self.out_coo_tensor, self.cut_io_mask, self.pos_io_channel

    @tf.function
    def get_output_cell_states(self, x):
        """Returns the values of all the idden channels of the output cells, organized according to the batch"""
        return tf.reshape(tf.gather_nd(x, self.out_coo_tensor), (self.batch_size, self.nb_out_cells, self.channel_n))


    @tf.function
    def call(self, x, updates=None, custom_channel_value=None, step_size=1.0):
        return self.get_next_state(x, self.nb_in_cells, self.cut_io_mask, self.pos_io_channel,
                              self.in_coo_tensor, updates,custom_channel_value, self.fire_rate,step_size)
       #we define the auxilliaury "get_next_state" function to be able to change dinamically the
       #input and output position without the need for tf retracing

    @tf.function
    def get_next_state(self, x,nb_in_cells,cut_io_mask,pos_io_channel,in_coo_tensor,
                       updates=None,custom_channel_value=None, fire_rate=None,step_size=1.0):


        if custom_channel_value is not None:
            value_r = tf.reshape(custom_channel_value, (self.size,self.size,1))
            x = x*self.chan_mask_neg + value_r*self.chan_mask

        if self.torus_boundaries:
            x_pad = cyclic_padding(x, 1)
        else:
            x_pad = x

        pre_life_mask = get_living_mask(x)
        dx = self.dmodel(x_pad)*step_size
        energy = x[:, :, :, 1:2]

        if self.energy_as_amplitude:
            energy_dx = dx[:,:,:,1:2]
            dx = (tf.math.sigmoid(dx) - 0.5)*double_ramp(energy)*2*self.max_amplitude
            if not(self.bound_energy_change):
                dx = dx*self.energy_mask_neg + energy_dx*self.energy_mask
                #if the enrgy update is not affected by the energy value
        elif self.max_amplitude is not None:
            dx = (tf.math.sigmoid(dx) - 0.5)*2*self.max_amplitude

        uniform_update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        uniform_update_mask = tf.cast(uniform_update_mask, tf.float32)

        if self.add_noise:
            dx += 0.01*tf.random.uniform(dx.shape, -1., 1.)

        if self.use_per_cell_fire_rate:

            default_fr_dead_cells = tf.cast(energy <= 0, tf.float32)*tf.nn.max_pool2d(energy, 3, [1, 1, 1, 1], 'SAME')
            #the dead cells get the fr of their most excited neighbor

            energy_update_mask =  (energy + default_fr_dead_cells - tf.random.uniform(tf.shape(x[:, :, :, :1]))) >= 0
            energy_update_mask = tf.cast(energy_update_mask, tf.float32)
            #the energy mask dictate the fire rate of each individual cells. The minimal fire rate
            # for a living or neighbor of a living cell is 0.2
            x += dx * uniform_update_mask * energy_update_mask
        else:
            x += dx * uniform_update_mask

        #x = dx * update_mask  + (tf.ones(tf.shape(x[:, :, :, :1])) - update_mask)*x
        post_life_mask = get_living_mask(x)
        life_mask =  post_life_mask & pre_life_mask
        life_mask = tf.cast(life_mask, tf.float32)
        if not(self.all_cells_alive):
            x = x*life_mask

        x = x * self.cut_io_mask + self.pos_io_channel
        if self.custom_in_signal:
            updates = tf.reshape(updates, tf.constant([self.batch_size*nb_in_cells]))
            input_cells_update = tf.scatter_nd(self.in_coo_tensor, updates, self.shape)
            #updates must be a tensor of shape (batch_size,nb of input cells). Each batch_size consecutive values
            #are the next values of the input cells of the corresponding CA in the batch
            if self.use_hidden_inputs:
                input_cells_update =  input_cells_update*self.hidden_inputs_mask #we remove the update of the hidden inputs
            x = x + input_cells_update

        if self.bound_values:
            if not(self.bound_energy_change):
                energy_clipped = tf.clip_by_value(x, -2, 2)
            x_clip = tf.clip_by_value(x, self.value_bounds[0], self.value_bounds[1])
            x=x_clip
            #the energy channel is designed to be always  between -2 and 2
            if not(self.bound_energy_change):
                x = x_clip*self.energy_mask_neg + energy_clipped*self.energy_mask

        if self.enforce_connexity:
            cell_pos = x[:,:,:,1:2] > 0.1
            cell_pos = tf.cast(cell_pos, tf.float32)
            connexity_mask = get_connected_cells(cell_pos, x[:,:,:,2:3], self.size)
            x*= connexity_mask
        if not(custom_channel_value is None):
            x = x*self.chan_mask_neg + value_r*self.chan_mask

        return x

    def visualize_ca(self,init_state, nb_step=30,
                    input_seq=None,
                    frames=None, dt=0.01,
                    init_wait=0.01, channels=[0,1],
                    custom_channel_val=None,
                    custom_bounds={}, verbose=True,
                    recreate_fig_each_step=False,titles={},
                    output_video=False, img_folder="video_images/",
                    video_name="CA_video", fps=6, dpi=150, form="png",
                    img_step=1):
        """Visualise the neural CA dynamics of the desired channels when predicting
        an output on a given set of inputs.
        If your are not in an environment allowing for interactive plotting,
        use recreate_fig_each_step=True, this will delete and recreate the figure
        at each step. It's slower but it works."""
        #we clean the previous images
        if output_video:
            if not(os.path.isdir(img_folder)):
                os.mkdir(img_folder)
            video_name += get_random_id() #we add a random suffix to avoid accidental deletion
            images = [img_folder+img for img in os.listdir(img_folder)]
            for img in images:
                os.remove(img)

        x = init_state
        if frames is None:
            frames= []
            frames.append(x[0,:,:])
            for i in range(nb_step):
                if verbose:
                    print("\rProgress: %.2f %%"%(i*100/nb_step), end='')
                if input_seq is None and custom_channel_val is None:
                    x = self(x)
                elif input_seq is not None and custom_channel_val is None:
                    x = self(x, updates=input_seq[:,i])
                elif input_seq is not None and custom_channel_val is None:
                    x = self(x, updates=input_seq[:,i])
                elif input_seq is None and custom_channel_val is not None:
                    x = self(x,custom_channel_value=custom_channel_val)
                elif input_seq is not None and custom_channel_val is not None:
                    x = self(x,custom_channel_value=custom_channel_val,
                                                updates=input_seq[:,i])

                x_np = x.numpy()
                frames.append(x_np[0,:,:])
            frames=np.array(frames)
        else:
            nb_step = len(frames)

        nb_chan = len(channels)

        for i in range(nb_step):
            if recreate_fig_each_step and i%img_step!=0:
                continue

            if i == 0 or recreate_fig_each_step:
                if recreate_fig_each_step:
                    clear_output(wait=True)

                fig2, axes = plt.subplots(nb_chan//2,2,figsize=(10,10))
                axes = axes.flatten()
                imgs = []
                channel_maxs = []
                channel_mins = []
                for c in range(len(channels)):
                    cmap_type = 'bwr'
                    channel_max = np.max(frames[:,:,:,channels[c]])
                    channel_min = np.min(frames[:,:,:,channels[c]])
                    channel_maxs.append(channel_max)
                    channel_mins.append(channel_min)

                    if channels[c] in custom_bounds:
                        min_val = custom_bounds[channels[c]][0]
                        max_val = custom_bounds[channels[c]][1]
                    else:
                        min_val = channel_min
                        max_val = channel_max



                    img = axes[c].matshow(frames[i,:,:,channels[c]], vmin=min_val, vmax=max_val, cmap=cmap_type)
                    imgs.append(img)

                    if channels[c] in titles:
                        chan_name = titles[channels[c]]
                    else:
                        chan_name = "Channel #"+str(channels[c])
                    axes[c].set_title("Frame #"+str(i)+"\n "+chan_name+ \
                                      "\nMin value: "+str(channel_min)[:4]+" Max value: "+str(channel_max)[:4] )
                    for x,y in self.in_cells:
                        for c in range(nb_chan):
                            axes[c].scatter( y,x,marker="x",  color="g")
                    for x,y in self.out_cells:
                        for c in range(nb_chan):
                            axes[c].scatter( y,x,marker="x",  color="orange")
                    fig2.tight_layout()
                plt.pause(init_wait)
                fig2.colorbar(img)
            else:
                for c in range(len(channels)):
                    imgs[c].set_data(frames[i,:,:,channels[c]])

                    if channels[c] in titles:
                        chan_name = titles[channels[c]]
                    else:
                        chan_name = "Channel #"+str(channels[c])
                    axes[c].set_title("Frame #"+str(i)+"\n "+chan_name+\
                                      "\nMin value: "+str(channel_mins[c])[:4]+" Max value: "+str(channel_maxs[c])[:4])
                plt.pause(dt) # pause avec duree en secondes
                if output_video:
                    fig2.savefig(img_folder+"/img_"+str(i)+"."+form,format=form, dpi=dpi, optimize=True)

            plt.show()
        if output_video:
            save_to_video(video_name,img_folder,fps,form)
        return frames

    def plot_losses(self, beg=0, avg_on=20, high_bound=None):
        """Plot the loss history of the training procedure.
        beg --- the beginning step to start the plot
        avg_on --- the width of the window on wich compute the moving average
        high_bound --- if not None, this number is used to clip loss
                        values that are above"""
        losses = np.array(self.loss_history[beg:])
        m= np.min(losses)

        if high_bound is not None:
            losses = np.clip(losses, m, high_bound)

        N = len(losses)

        if m <0:
            log_losses = np.log10(losses-m +1)
        else:
            log_losses = np.log10(losses+1e-8)
        rolling_avg_losses = get_rolling_avg(losses, k=avg_on)
        rolling_avg_log_losses = get_rolling_avg(log_losses, k=avg_on)

        rol_avg_X = np.arange(avg_on//2, N-avg_on+avg_on//2+1)

        fig, axes = plt.subplots(2,1,figsize=(12,20))
        axes = axes.flatten()
        axes[0].plot(losses, label="Loss", marker='x',alpha=0.7, linestyle='none')
        axes[0].plot(rol_avg_X,rolling_avg_losses, label="Rolling average on "+str(avg_on)+" steps of the loss")
        axes[0].set_xlabel("Training step")
        axes[0].set_title("Evolution of the loss")
        axes[0].legend()

        axes[1].plot(log_losses, label="Log10 of loss", marker='x', linestyle='none')
        axes[1].plot(rol_avg_X,rolling_avg_log_losses, label="Rolling average on "+str(avg_on)+" steps of the log10 loss")
        axes[1].set_xlabel("Training step")
        axes[1].set_title("Evolution of the loss (log scale)")
        axes[1].legend()
        plt.show()













