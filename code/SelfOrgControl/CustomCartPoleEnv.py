"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnvContiReward(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is cos(x*pi/(2L))² for every step taken, -100 for the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, cyclic_boundary=False, auto_reset=True,
                    show_arrows=True):
        """
            cyclic_boundary -- If true, hitting a wall on one side teleport the
                                cart to the other side.
            auto_reset -- If true, the environment automatically reset if the cart
                            hit a wall of the pole fall, without
                            causing termination.
            show_arrows -- plot arrow showing angular and linear velocity in the
                            render.
        """
        self.cyclic_boundary = cyclic_boundary
        self.show_arrows = show_arrows
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.auto_reset = auto_reset



        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self._max_episode_steps = 10000

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.step_nb = 0
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.step_nb +=1


        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        if self.cyclic_boundary:
            if x > self.x_threshold:
                x = -self.x_threshold + 0.1

            if x < -self.x_threshold:
                x = self.x_threshold - 0.1

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.step_nb  > self._max_episode_steps
        )

        if done and self.auto_reset:
            self.reset()
            done =False

        reward = 0.0
        if not done:
            reward = 1.0
        if self.steps_beyond_done is None and done:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
                self.steps_beyond_done += 1
                reward = 0.0

        # reward_theta = (np.cos(theta)+1.0)/2.0
        # reward_x = np.cos((x/self.x_threshold)*(np.pi/2))
        #
        # reward = reward_theta*reward_x
        reward = np.cos((x/self.x_threshold)*(np.pi/2))**2

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.step_nb = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def set_state(self, new_state, pole_len=None):
        self.steps_beyond_done
        self.state = new_state

        if pole_len is not None:
            self.length = pole_len  # actually half the pole's length
            self.polemass_length = (self.masspole * self.length)


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        #Arrow render param

        pad = 7.0
        self.vel_arrow_len = 0
        arrowwidth = 10.0
        tipwidth = 20.0

        self.theta = 0

        # action taken

        pad_act = 200


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cart.set_color(0.0, 0.0, 0.0)
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.cart = cart
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)



            if self.show_arrows:
                #linear velocity arrow
                l, r, t, b = cartwidth + pad, cartwidth + pad+ self.vel_arrow_len, arrowwidth/2, -arrowwidth/2

                lin_arrow_body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                lin_arrow_body.set_color(0.7, 0.7, 0.7)
                lin_arrow_body.add_attr(self.carttrans)

                self.lin_arrow_body =lin_arrow_body

                self.viewer.add_geom(lin_arrow_body)

                t,b, tip = tipwidth/2, -tipwidth/2, np.sqrt(3)*tipwidth/2

                lin_arrow_tip = rendering.FilledPolygon([(r, t), (r+tip, 0), (r, b)])
                lin_arrow_tip.set_color(0.7, 0.7, 0.7)
                lin_arrow_tip.add_attr(self.carttrans)
                self.lin_arrow_tip = lin_arrow_tip
                self.viewer.add_geom(lin_arrow_tip)

                #angular velocity arrow body
                ang_arrow_len = 0
                theta = 0

                od = (np.cos(theta), -np.sin(theta) ) #direction othogonal to the pole
                pd = (np.sin(theta), np.cos(theta) ) #pole direciton
                L = polelen - polewidth / 2
                pt = (L*np.sin(theta), L*np.cos(theta)) #pole tip

                ang_arrow_body = rendering.FilledPolygon([(pt[0]+pad*od[0]+(arrowwidth/2)*pd[0],
                                                            pt[1]+pad*od[1]+(arrowwidth/2)*pd[1]),
                                                        (pt[0]+(pad+ang_arrow_len)*od[0]+(arrowwidth/2)*pd[0],
                                                            pt[1]+(pad+ang_arrow_len)*od[1]+(arrowwidth/2)*pd[1]),
                                                        (pt[0]+(pad+ang_arrow_len)*od[0]-(arrowwidth/2)*pd[0],
                                                            pt[1]+(pad+ang_arrow_len)*od[1]-(arrowwidth/2)*pd[1]),
                                                        (pt[0]+pad*od[0]-(arrowwidth/2)*pd[0],
                                                            pt[1]+pad*od[1]-(arrowwidth/2)*pd[1])])
                ang_arrow_body.set_color(0.7, 0.7, 0.7)
                ang_arrow_body.add_attr(self.carttrans)
                self.ang_arrow_body =ang_arrow_body
                self.viewer.add_geom(ang_arrow_body)

                #angular velocity arrow tip
                ang_arrow_tip = rendering.FilledPolygon([(pt[0]+(pad+ang_arrow_len)*od[0]+(tipwidth/2)*pd[0],
                                                            pt[1]+(pad+ang_arrow_len)*od[1]+(tipwidth/2)*pd[1]),
                                                        (pt[0]+(pad+ang_arrow_len)*od[0]-(tipwidth/2)*pd[0],
                                                            pt[1]+(pad+ang_arrow_len)*od[1]-(tipwidth/2)*pd[1]),
                                                        (pt[0]+(pad+ang_arrow_len+np.sqrt(3)*tipwidth/2)*od[0],
                                                            pt[1]+(pad+ang_arrow_len+np.sqrt(3)*tipwidth/2)*od[1])])
                ang_arrow_tip.set_color(0.7, 0.7, 0.7)
                ang_arrow_tip.add_attr(self.carttrans)
                self.ang_arrow_tip = ang_arrow_tip
                self.viewer.add_geom(ang_arrow_tip)

            #rest of the environment
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)


            self._pole_geom = pole

        if self.state is None:
            return None

        x = self.state

        #Edit the arrow geometry
        if self.show_arrows:
            #linear velocity arrow
            self.vel_arrow_len = abs(x[1]*40.0)
            #side = int(self.vel_arrow_len >0)

            if x[1]>0:
                l, r, t, b = cartwidth/2 + pad, cartwidth/2 + pad+ self.vel_arrow_len, arrowwidth/2, -arrowwidth/2
                tip_top,tip_bot, tip_end = tipwidth/2, -tipwidth/2, np.sqrt(3)*tipwidth/2
                self.lin_arrow_tip.v = [(r, tip_top), (r+tip_end, 0), (r, tip_bot)]

            else:
                l,r,t,b = -cartwidth/2 -pad -self.vel_arrow_len, -cartwidth/2 -pad, arrowwidth/2, -arrowwidth/2
                tip_top,tip_bot, tip_end = tipwidth/2, -tipwidth/2, np.sqrt(3)*tipwidth/2
                self.lin_arrow_tip.v = [(l, tip_top), (l-tip_end, 0), (l, tip_bot)]

            self.lin_arrow_body.v = [(l, b), (l, t), (r, t), (r, b)]

            #angular velocity arrow
            pad += polewidth/2
            ang_arrow_len = abs(x[3]*40.0)
            theta = x[2]

            od = (np.cos(theta), -np.sin(theta) ) #direction othogonal to the pole
            pd = (np.sin(theta), np.cos(theta) ) #pole direciton
            L = polelen - polewidth / 2
            pt = (L*np.sin(theta), L*np.cos(theta)) #pole tip
            if x[3] > 0:
                self.ang_arrow_body.v = [(pt[0]+pad*od[0]+(arrowwidth/2)*pd[0],
                                            pt[1]+pad*od[1]+(arrowwidth/2)*pd[1]),
                                        (pt[0]+(pad+ang_arrow_len)*od[0]+(arrowwidth/2)*pd[0],
                                            pt[1]+(pad+ang_arrow_len)*od[1]+(arrowwidth/2)*pd[1]),
                                        (pt[0]+(pad+ang_arrow_len)*od[0]-(arrowwidth/2)*pd[0],
                                            pt[1]+(pad+ang_arrow_len)*od[1]-(arrowwidth/2)*pd[1]),
                                        (pt[0]+pad*od[0]-(arrowwidth/2)*pd[0],
                                            pt[1]+pad*od[1]-(arrowwidth/2)*pd[1])]
                self.ang_arrow_tip.v = [(pt[0]+(pad+ang_arrow_len)*od[0]+(tipwidth/2)*pd[0],
                                            pt[1]+(pad+ang_arrow_len)*od[1]+(tipwidth/2)*pd[1]),
                                        (pt[0]+(pad+ang_arrow_len)*od[0]-(tipwidth/2)*pd[0],
                                            pt[1]+(pad+ang_arrow_len)*od[1]-(tipwidth/2)*pd[1]),
                                        (pt[0]+(pad+ang_arrow_len+np.sqrt(3)*tipwidth/2)*od[0],
                                            pt[1]+(pad+ang_arrow_len+np.sqrt(3)*tipwidth/2)*od[1])]
            else:
                self.ang_arrow_body.v = [(pt[0]-pad*od[0]+(arrowwidth/2)*pd[0],
                                            pt[1]-pad*od[1]+(arrowwidth/2)*pd[1]),
                                        (pt[0]-(pad+ang_arrow_len)*od[0]+(arrowwidth/2)*pd[0],
                                            pt[1]-(pad+ang_arrow_len)*od[1]+(arrowwidth/2)*pd[1]),
                                        (pt[0]-(pad+ang_arrow_len)*od[0]-(arrowwidth/2)*pd[0],
                                            pt[1]-(pad+ang_arrow_len)*od[1]-(arrowwidth/2)*pd[1]),
                                        (pt[0]-pad*od[0]-(arrowwidth/2)*pd[0],
                                            pt[1]-pad*od[1]-(arrowwidth/2)*pd[1])]
                self.ang_arrow_tip.v = [(pt[0]-(pad+ang_arrow_len)*od[0]+(tipwidth/2)*pd[0],
                                            pt[1]-(pad+ang_arrow_len)*od[1]+(tipwidth/2)*pd[1]),
                                        (pt[0]-(pad+ang_arrow_len)*od[0]-(tipwidth/2)*pd[0],
                                            pt[1]-(pad+ang_arrow_len)*od[1]-(tipwidth/2)*pd[1]),
                                        (pt[0]-(pad+ang_arrow_len+np.sqrt(3)*tipwidth/2)*od[0],
                                            pt[1]-(pad+ang_arrow_len+np.sqrt(3)*tipwidth/2)*od[1])]

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]


        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        if self.steps_beyond_done is not None:
            pole.set_color(.8, .2, .2)
            self.cart.set_color(.3, 0.1, 0.1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
