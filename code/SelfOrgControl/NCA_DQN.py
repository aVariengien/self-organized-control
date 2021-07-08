import os

import random
import gym
from SelfOrgControl.CustomCartPoleEnv import CartPoleEnvContiReward
import numpy as np
import tensorflow as tf
from collections import deque
from SelfOrgControl.NeuralCA import *
from IPython import display as ipythondisplay

# The original structure of the code was taken from
#  https://github.com/pythonlessons/Reinforcement_Learning/tree/master/01_CartPole-reinforcement-learning



class DQNAgent:
    def __init__(self, cyclic_boundary=False, auto_reset = False,
                        show_arrows=True):
        self.env = CartPoleEnvContiReward(cyclic_boundary=cyclic_boundary,
                                          auto_reset=auto_reset,
                                          show_arrows=show_arrows)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000000
        self.max_len_memory = 50000

        self.memory = deque(maxlen=self.max_len_memory)

        self.use_batch_training = True

        self.gamma = 0.95    # discount rate
        self.epsilon = 1. # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 128
        self.train_start = 1000 #number of environment step before begin of the
                                #training
        self.nb_replay = 0

        self.in_factors = np.array([2., 0.25, 4., 0.15])
        self.out_factor = 0.01

        self.file_name= "cartpole-dqn-NeuralCA_replay_"

        self.cur_state = None #for step by step test to create visualisation

        # create main model
        lr = 5e-3
        lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [1000,10000], [lr, lr*0.1, lr*0.001])

        inp_cell_pos = [(11, 26),(25,20),(5,20),(19,6),(19,26),(5,13),(25,12) ,(11, 6)]
        out_cell_pos = [(13,16), (17,16)]
        self.model = TrainableNeuralCA(input_electrodes = inp_cell_pos,
                                    output_electrodes = out_cell_pos,
                                    grid_size=32,
                                    batch_size=16, channel_n=6,
                                    ca_steps_per_sample=(50,60),
                                    replace_proba=0.01,
                                    task_loss_w=0.5, grid_pool_size=100,
                                    learning_rate=lr,
                                    repeat_input=2,torus_boundaries=False,
                                    penalize_overflow=True, overflow_w = 1e2,
                                    use_hidden_inputs=True, perturb_io_pos=True,
                                    add_noise=False, damage=True,
                                    nb_hid_range=(0,0), move_rad=0, proba_move=0.0)

        #the pool of cart-pole states
        self.cartpole_states = [ self.env.reset() for k in range(10)]
        self.max_env_it = 2

        self.longest_run = 0
        self.losses = []
        self.best_model_name = ""



    def remember(self, state, action, reward, next_state, done):
        experience = {"state":state, "action": action, "reward": reward,
                     "next_state":next_state, "done":done }

        self.memory.append(experience)

        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state*self.in_factors))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory

        minibatch = random.choices(self.memory, k=min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i]["state"]
            action.append(minibatch[i]["action"])
            reward.append(minibatch[i]["reward"])
            next_state[i] = minibatch[i]["next_state"]
            done.append(minibatch[i]["done"])

        predictions = self.model.predict(state*self.in_factors, use_batch=True)/ self.out_factor
        target = predictions.copy()
        target_next = self.model.predict(next_state*self.in_factors, use_batch=True)/ self.out_factor


        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))


        # We divide the batch of experiences of size self.batch_size in
        # self.batch_size//self.model.batch_size smaller batch of
        # size self.model.batch_size (the batch size of the neural CA)

        if self.use_batch_training:
            state = np.reshape(state, (self.batch_size//self.model.batch_size,
                                self.model.batch_size, self.state_size) )

            target = np.reshape(target, (self.batch_size//self.model.batch_size,
                                        self.model.batch_size, 2) )

        # we scale the inputs and outputs according to the predefined factors
        state = state*self.in_factors
        target = target*self.out_factor

        loss = self.model.fit(state, target, verbose=False,
                        use_batch=self.use_batch_training)
        self.losses.append(loss)

        if self.nb_replay%10 == 0:

            sc,_, nb_t,_ = self.test(render=False, verbose=False)
            if self.longest_run < nb_t:
                filename_save = self.file_name+str(self.nb_replay)+"_score_"+\
                                str(np.round(sc,4))+"_"+str(nb_t) +"_it"
                self.best_model_name = filename_save
                print("New high score, model saved to "+filename_save)
                self.model.neuralCA.dmodel.save_weights(filename_save)
                self.longest_run = nb_t

            print(("\rReplay #%.i | Epsilon: %.3f | log10 loss: %.2f " +\
                    "| Test score: %.2f | Test duration: %.i | Best duration: %.i        ")%(self.nb_replay,
                                                                self.epsilon,
                                                                 np.log10(np.mean(self.losses)),
                                                                  sc, nb_t, self.longest_run),
                                                                  end='')
            self.losses = []

        self.EPISODES -= 1
        self.nb_replay +=1

    def load(self, name):
        self.model.load(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        """Run the training process of the DQN agent"""
        while self.EPISODES>0:

            # Sampling from the pool of cart-pole states
            rd_id = random.randint(0,len(self.cartpole_states)-1)
            state= self.cartpole_states[rd_id]
            self.env.reset()
            self.env.set_state(state)
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            tot_score = 0

            # K iteration of the environment
            while not done and i<self.max_env_it:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                tot_score += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

            if done:
                self.cartpole_states[rd_id] = np.squeeze(self.env.reset())
            else:
                self.cartpole_states[rd_id] = np.squeeze(state)

            self.replay()



    def test_one_step(self, render=True, fix_nb_step=-1, conti_damage_proba=None):
        """Advance the test of the DQN agent of one step and return the
            frame."""
        if self.cur_state is None:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            self.cur_state = state
        if self.cur_state == "END":
            return self.prev_img



        if conti_damage_proba is not None:
            pred, _ = self.model.predict(self.cur_state*self.in_factors,
                        conti_damage_proba=conti_damage_proba,
                        return_all_grids=True, no_reinit=True)
        else:
            pred = self.model.predict(self.cur_state*self.in_factors,
                        conti_damage_proba=conti_damage_proba,
                        return_all_grids=False)

        action = np.argmax(pred)

        next_state, reward, done, _ = self.env.step(action)
        state = np.reshape(next_state, [1, self.state_size])
        self.cur_state = state
        if done:
            self.cur_state = "END"

        if render:
            img = self.env.render(mode="rgb_array")
            self.prev_img = img

        return img

    def test(self, render=True, nb_episode=1, fix_nb_step=-1,
                random_action_proba=False, verbose=1,
                conti_damage_proba=None, return_sensors=False,
                render_for_colab=False):
        """Test the agent for nb_episodes runs. Returns the mean and standard
            deviation of the lengths of the episodes and the score.
            verbose -- 0: no information printed
                       1: results of each episode printed
                       2: action and observation at each step of the episode

            fix_nb_step -- If different of -1, this value is used to fix a limit
                            of the lengths of the simulations.

            random_action_proba -- The proba of taking a random action at each
                                    step

            conti_damage_proba -- If not none, this value is used as the proba of
                                    damaging the grid at each step
            return_sensors -- Whether to return the list of observations"""

        if verbose>0:
            print("----------------- Begin testing -----------------")
        ACTIONS = ["left", "right"]
        S = []
        mean_it = []
        sensors = []
        for e in range(nb_episode):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            tot_score = 0

            while ((not done and fix_nb_step ==-1) or
                  (i<fix_nb_step and fix_nb_step !=-1)):
                if render:
                    self.env.render()
                if render_for_colab:
                    screen = self.env.render(mode='rgb_array')
                    plt.imshow(screen)
                    plt.axis("off")
                    ipythondisplay.clear_output(wait=True)
                    ipythondisplay.display(plt.gcf())

                if conti_damage_proba is not None:
                    pred, _ = self.model.predict(state*self.in_factors,
                                conti_damage_proba=conti_damage_proba,
                                return_all_grids=True, no_reinit=True)
                else:
                    pred = self.model.predict(state*self.in_factors,
                                conti_damage_proba=conti_damage_proba,
                                return_all_grids=False)
                action = np.argmax(pred)
                if verbose == 2:
                    print(ACTIONS[action])
                    print(pred/self.out_factor)
                    print(state*self.in_factors)

                if random_action_proba is not None:
                    if np.random.random() < random_action_proba:
                        action=np.random.randint(0,2)

                next_state, reward, done, _ = self.env.step(action)
                if return_sensors:
                    sensors.append(next_state*self.in_factors)
                tot_score += reward
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done and fix_nb_step==-1 and verbose>0:
                    print("Test #{}, score: {:.5}, duration: {}".format(e, tot_score,i))
            S.append(tot_score)
            mean_it.append(i)
        self.env.close()
        if return_sensors:
            return np.mean(S),np.std(S), np.mean(mean_it), np.std(mean_it), sensors
        else:
            return np.mean(S),np.std(S), np.mean(mean_it), np.std(mean_it)



