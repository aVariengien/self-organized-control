import numpy as np
import tensorflow as tf


## Function used in the additional experiments

def create_distance_mask(output_coo,N, shape="square"):
    if shape=="square":
        d = np.zeros((1,N,N,1), "float32")
        d[0][output_coo[0]][output_coo[1]][0] = 1.
        d = tf.convert_to_tensor(d)
        for t in range(N):
            m = tf.nn.max_pool2d(d, 3, [1, 1, 1, 1], "SAME")
            m = tf.cast(m>0, tf.float32)
            #we add a little amount of noise such that each value in d will be unique
            #this becomes useful when we want to easily recover *one* of the living cells the colsest
            #to the output.
            m*= tf.random.uniform((1,N,N,1))*0.001 + 0.9995
            d +=m
        d = tf.reshape(d, (N,N))
        return d

    elif shape=="circle":
        d = np.zeros((N,N), "float32")
        x_out, y_out = output_coo[0], output_coo[1]
        for x in range(N):
            for y in range(N):
                d[x][y] = np.sqrt((x-x_out)**2 + (y-y_out)**2)
        d = -d
        d += -np.min(d)
        d *= np.random.random((N,N))*0.001 + 0.9995
        d = tf.convert_to_tensor(d)
        return d


@tf.function
def loss_distance_to_output(x, distance_mask,batch_size=8,
                            energy_budget=10, k=10, return_array=False):
    energy = x[:,:,:,1]
    non_zero = tf.cast(energy != 0, tf.float32)
    max = tf.reduce_max(non_zero*distance_mask, axis=[1,2])
    max = tf.reshape(max, (batch_size,1,1))

    max_pos = tf.cast(tf.math.equal(distance_mask,max), tf.float32)
    energy_cliped = -tf.nn.relu(1-energy) +1
    proximity_score = tf.reduce_sum(max_pos*energy_cliped*distance_mask, axis=[1,2])

    energy_cost = tf.nn.relu(tf.math.reduce_sum(tf.nn.relu(energy), axis=[1,2]) - energy_budget)

    #the proximity score is subject to maximization hence the -k factor
    batch_losses = -k*proximity_score + energy_cost

    if return_array:
        return batch_losses, proximity_score, energy_cost
    else:
        return tf.math.reduce_sum(batch_losses)


@tf.function
def train_step(nca, x, trainer, distance_mask=None,custom_channel=False,
               custom_channel_val=None,input_seq=None,target_seq=None,
               min_step=30, max_step=50, energy_budget=10,
               task_w = 5):
    """ A function for training neural CA on other tasks that the ones
        handled by the TrainableNeuralCA class"""

    iter_n = tf.random.uniform([], min_step, max_step, tf.int32)
    losses_sum = tf.zeros([], tf.float32 )
    with tf.GradientTape() as g:
        for i in tf.range(iter_n):
            if (input_seq is not None) and not(custom_channel):
                x = nca(x, updates = input_seq[:,i])
            elif (input_seq is not None) and custom_channel:
                x = nca(x,updates = input_seq[:,i], custom_channel_value=custom_channel_val)
            elif (input_seq is None) and custom_channel:
                x = nca(x, custom_channel_value=custom_channel_val)
            else:
                x=nca(x)

        loss= loss_distance_to_output(x, distance_mask,energy_budget=energy_budget,
                                                k=task_w, batch_size=nca.batch_size)

    grads = g.gradient(loss, nca.dmodel.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, nca.dmodel.weights))
    return x, loss