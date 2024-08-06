from __future__ import print_function
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
# from Src.Utils.utils import Space, binaryEncoding
import gym
import cv2
import time
def binaryEncoding(num, size):
    binary = np.zeros(size)
    i = -1
    while num > 0:
        binary[i] = num % 2
        num = num//2
        i -= 1
    return binary

class Gridworld_CL(gym.core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_actions=4,
                 debug=False,
                 max_step_length=0.2,
                 max_steps=30,
                 im_size = 100,
                 image_output = True,
                 simplify_action = False,
                 max_episodes=1e5,
                 change_interval=-1,
                 change_count=3):

        self.debug = debug
        self.ep_count = 0


        # General parameters for the environemnt
        self.n_actions = n_actions
        self.im_size = im_size
        self.action_space = gym.spaces.Discrete(2**n_actions)#Space(size=2**n_actions)
        # self.observation_space = gym.spaces.Box(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)
        n_channels = 3
        if image_output:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.im_size, self.im_size, n_channels),
                                            dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)

        self.disp_flag = False

        self.movement = self.get_movements(self.n_actions)
        self.motions = self.get_action_motions(self.n_actions)

        if simplify_action:
            unique_motion = np.unique(self.motions,axis=0)
            unique_action_num = len(unique_motion)
            self.motions = unique_motion
            self.action_space = gym.spaces.Discrete(unique_action_num)
        # print("action_space:")
        # print(self.action_space)
        self.wall_width = 0.05
        self.step_unit = self.wall_width - 0.005
        self.repeat = int(max_step_length / self.step_unit)

        self.max_steps = int(max_steps / max_step_length)
        self.step_reward = -0.05
        self.collision_reward = 0  # -0.05
        self.movement_reward = 0  # 1
        self.randomness = 0

        self.gird_width = max_step_length / 10
        self.pic_size = int(1 / self.gird_width)

        self.image_output = image_output
        self.viewer = None


        self.n_lidar = 0
        self.angles = np.linspace(0, 2 * np.pi, self.n_lidar + 1)[:-1]  # Get 10 lidar directions,(11th and 0th are same)
        # self.lidar_angles = np.array(list(zip(np.cos(self.angles), np.sin(self.angles))), dtype=np.float32)
        self.lidar_angles = list(zip(np.cos(self.angles), np.sin(self.angles)))
        self.static_obstacles = self.get_static_obstacles()

        # Continual Learning parameters
        # self.rng = np.random.RandomState(0)
        # if change_interval > 0:
        #     self.change_interval = change_interval
        # else:
        #     self.change_interval = max_episodes // change_count
        #
        # self.change_add_count = int(1.0/change_count * 2**self.n_actions)
        # self.action_mask = np.zeros(2**n_actions)  #  Mask to indicate the currently active set of actions
        # self.action_tracker = np.zeros(2**n_actions)  # Tracks all the actions that have been made available till now

        if debug:
            self.heatmap_scale = 99
            self.heatmap = np.zeros((self.heatmap_scale + 1, self.heatmap_scale + 1))

        self.reset()

    def seed(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def get_embeddings(self):
        return self.motions.copy()

    def pos_convert_discrete(self, x):
        discrete_x = int(x / self.gird_width)
        discrete_x = np.clip(discrete_x,0,self.pic_size-1)
        return discrete_x

    def _obs_from_state(self,image_output):
        if image_output:
            dot_size = 1 * self.gird_width
            state_grid = np.zeros((self.pic_size, self.pic_size))
            x1 = self.pos_convert_discrete(self.curr_state[0] - dot_size)
            x2 = self.pos_convert_discrete(self.curr_state[0] + dot_size)

            y1 = self.pos_convert_discrete(self.curr_state[1] - dot_size)
            y2 = self.pos_convert_discrete(self.curr_state[1] + dot_size)

            for ii in range(x1, x2):
                for jj in range(y1, y2):
                    state_grid[ii,jj] = 1

            goal_grid = np.zeros((self.pic_size, self.pic_size))
            for key, val in self.reward_states.items():
                coords, cond = val
                if cond:
                    x1, y1, x2, y2 = coords
                    x1 = self.pos_convert_discrete(x1)
                    x2 = self.pos_convert_discrete(x2)
                    y1 = self.pos_convert_discrete(y1)
                    y2 = self.pos_convert_discrete(y2)
                    for ii in range(x1, x2):
                        for jj in range(y1, y2):
                            goal_grid[ii, jj] = 1

            map_grid = np.zeros((self.pic_size, self.pic_size))
            for coords in self.static_obstacles:
                x1, y1, x2, y2 = coords
                x1 = self.pos_convert_discrete(x1)
                x2 = self.pos_convert_discrete(x2)
                y1 = self.pos_convert_discrete(y1)
                y2 = self.pos_convert_discrete(y2)
                for ii in range(x1, x2):
                    for jj in range(y1, y2):
                        map_grid[ii, jj] = 1
            im_list = [state_grid, map_grid, goal_grid]
            im_stack = 70 * np.stack(im_list, axis=-1).astype(np.uint8)
            obs = cv2.resize(im_stack, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_AREA)
            return obs

        else:
            return self.curr_state

    def render(self, mode='human', img=None):
        if img is None:
            img = self._obs_from_state(True)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def render_original(self):
        x, y = self.curr_pos
        d = 0.02

        # ----------------- One Time Set-up --------------------------------
        if not self.disp_flag:
            self.disp_flag = True
            # plt.axis('off')
            self.currentAxis = plt.gca()
            plt.figure(1, frameon=False)                            #Turns off the the boundary padding
            self.currentAxis.xaxis.set_major_locator(NullLocator()) #Turns of ticks of x axis
            self.currentAxis.yaxis.set_major_locator(NullLocator()) #Turns of ticks of y axis
            plt.ion()                                               #To avoid display blockage

            self.circle = Circle((x, y), d, color='red')
            for coords in self.static_obstacles:
                x1, y1, x2, y2 = coords
                w, h = x2-x1, y2-y1
                self.currentAxis.add_patch(Rectangle((x1, y1), w, h, fill=True, color='gray'))
            print("Init done")
        # ----------------------------------------------------------------------

        for key, val in self.dynamic_obs.items():
            coords, cond = val
            if cond:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[key] = Rectangle((x1, y1), w, h, fill=True, color='black')
                self.currentAxis.add_patch(self.objects[key])


        for key, val in self.reward_states.items():
            coords, cond = val
            if cond:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[key] = Rectangle((x1, y1), w, h, fill=True)
                self.currentAxis.add_patch(self.objects[key])

        if len(self.angles) > 0:
            r = self.curr_state[-10:]
            coords = zip(r * np.cos(self.angles), r * np.sin(self.angles))

            for i, (w, h) in enumerate(coords):
                self.objects[str(i)] = Arrow(x, y, w, h, width=0.01, fill=True, color='lightgreen')
                self.currentAxis.add_patch(self.objects[str(i)])

        self.objects['circle'] = Circle((x, y), d, color='red')
        self.currentAxis.add_patch(self.objects['circle'])

        # remove all the dynamic objects
        plt.pause(1e-7)
        for _, item in self.objects.items():
            item.remove()
        self.objects = {}

    def set_rewards(self):
        # All rewards
        self.G1_reward = 100 #100
        self.G2_reward = 0

    def reset(self):
        """
        Sets the environment to default conditions
        :return: Starting state and the set of actions for the next trajectory
        """

        # change_flag = self.update_mask()  # Update the mask for next step onwards

        self.set_rewards()
        self.steps_taken = 0
        self.ep_count += 1
        self.reward_states = self.get_reward_states()
        self.dynamic_obs = self.get_dynamic_obstacles()
        self.objects = {}

        #x = 0.25
        #x = np.clip(x + np.random.randn()/30, 0.15, 0.35) # Add noise to initial x position
        self.curr_pos = np.array([0.25, 0.1])
        # self.curr_action = [0]*self.n_actions
        self.curr_state = self.make_state()

        return self._obs_from_state(image_output=self.image_output).copy() #, self.action_mask, change_flag

    def get_movements(self, n_actions):
        """
        Divides 360 degrees into n_actions and
        assigns how much it should make the agent move in both x,y directions

        usage:  delta_x, delta_y = np.dot(action, movement)
        :param n_actions:
        :return: x,y direction movements for each of the n_actions
        """
        x = np.linspace(0, 2*np.pi, n_actions+1)
        y = np.linspace(0, 2*np.pi, n_actions+1)
        motion_x = np.around(np.cos(x)[:-1], decimals=3)
        motion_y = np.around(np.sin(y)[:-1], decimals=3)
        movement = np.vstack((motion_x, motion_y)).T

        if self.debug: print("Movements:: ", movement)
        return movement

    def get_action_motions(self, n_actions):
        shape = (2**n_actions, 2)
        motions = np.zeros(shape)
        for idx in range(shape[0]):
            action = binaryEncoding(idx, n_actions)
            motions[idx] = np.dot(action, self.movement)

        # Normalize to make maximium distance covered at a step be 1
        max_dist = np.max(np.linalg.norm(motions, ord=2, axis=-1))
        motions /= max_dist

        return motions

    # def update_mask(self):
    #     if self.ep_count % self.change_interval == 0:
    #         add_actions = []
    #
    #         # add new actions that were never seen before
    #         curr_no_actions = np.where(self.action_tracker == 0)[0]
    #         if len(curr_no_actions) >= self.change_add_count:
    #                 l = len(curr_no_actions)
    #                 skip = int(l/self.change_add_count)
    #                 add_actions = curr_no_actions[np.arange(0, l, skip)]
    #         self.action_mask[add_actions] = 1
    #
    #         # Track the new actions added
    #         self.action_tracker[add_actions] = 1
    #
    #         print("Actions added: {}, Total actions now: {}".format(len(add_actions), sum(self.action_mask)))
    #
    #         return True
    #     return False

    def step(self, action):
        # assert self.action_mask[action]  # check if the action selected is in the current set of active actions

        self.steps_taken += 1
        reward = 0
        info = dict()
        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        term = self.is_terminal()
        if term:
            return self._obs_from_state(image_output=self.image_output).copy(), 0, term, info

        motion = self.motions[action]  # Table look up for the impact/effect of the selected action
        reward += self.step_reward
        prv_pos = self.curr_pos
        for i in range(self.repeat):
            if np.random.rand() < self.randomness:
                # Add noise some percentage of the time
                noise = np.random.rand(2)/1.415  # normalize by max L2 of noise
                delta = noise * self.step_unit  # Add noise some percentage of the time
            else:
                delta = motion * self.step_unit

            new_pos = self.curr_pos + delta  # Take a unit step in the direction of chosen action

            if self.valid_pos(new_pos):
                dist = np.linalg.norm(delta)
                reward += self.movement_reward * dist  # small reward for moving
                if dist >= self.wall_width:
                    print("ERROR: Step size bigger than wall width", new_pos, self.curr_pos, dist, delta, motion, self.step_unit)

                self.curr_pos = new_pos
                reward += self.get_goal_rewards(self.curr_pos)
                # reward += self.open_gate_condition(self.curr_pos)
            else:
                reward += self.collision_reward
                break

            # To avoid overshooting the goal

            if self.is_terminal():
                info['episode'] = {'r': reward, 'l': self.steps_taken}
                break

            # self.update_state()
            self.curr_state = self.make_state()

        if self.debug:
            # Track the positions being explored by the agent
            x_h, y_h = self.curr_pos*self.heatmap_scale
            self.heatmap[min(int(y_h), 99), min(int(x_h), 99)] += 1

            ## For visualizing obstacle crossing flaw, if any
            for alpha in np.linspace(0,1,10):
                mid = alpha*prv_pos + (1-alpha)*self.curr_pos
                mid *= self.heatmap_scale
                self.heatmap[min(int(mid[1]), 99)+1, min(int(mid[0]), 99)+1] = 1

        obs = self._obs_from_state(image_output=self.image_output).copy()
        # print(info)
        # print(self.curr_state)
        return obs, reward, self.is_terminal(), info


    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]

        # Append lidar values
        for cosine, sine in self.lidar_angles:
            r, r_prv = 0, 0
            pos = (x+r*cosine, y+r*sine)
            while self.valid_pos(pos) and r < 0.5:
                r_prv = r
                r += self.step_unit
                pos = (x+r*cosine, y+r*sine)
            state.append(r_prv)

        # Append the previous action chosen
        # state.extend(self.curr_action)

        return state

    def get_goal_rewards(self, pos):
        for key, val in self.reward_states.items():
            region, reward = val
            if reward and self.in_region(pos, region):
                self.reward_states[key] = (region, 0)  # remove reward once taken
                if self.debug: print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))

                return reward
        return 0

    def get_reward_states(self):
        self.G1 = (0.25, 0.45, 0.30, 0.5)
        self.G2 = (0.70, 0.85, 0.75, 0.90)
        return {'G1': (self.G1, self.G1_reward),
                'G2': (self.G2, self.G2_reward)}

    def get_dynamic_obstacles(self):
        """
        :return: dict of objects, where key = obstacle shape, val = on/off
        """
        return {}

        # self.Gate = (0.15,0.25,0.35,0.3)
        # return {'Gate': (self.Gate, self.Gate_reward)}

    def get_static_obstacles(self):
        """
        Each obstacle is a solid bar, represented by (x,y,x2,y2)
        representing bottom left and top right corners,
        in percentage relative to board size

        :return: list of objects
        """
        self.O1 = (0, 0.25, 0 + self.wall_width + 0.45, 0.25 + self.wall_width)  # (0, 0.25, 0.5, 0.3)
        self.O2 = (0.5, 0.25, 0.5 + self.wall_width, 0.25 + self.wall_width + 0.5)  # (0.5, 0.25, 0.55, 0.8)
        obstacles = [self.O1, self.O2]
        return obstacles

    def valid_pos(self, pos):
        flag = True

        # Check boundary conditions
        if not self.in_region(pos, [0,0,1,1]):
            flag = False

        # Check collision with static obstacles
        for region in self.static_obstacles:
            if self.in_region(pos, region):
                flag = False
                break

        # Check collision with dynamic obstacles
        for key, val in self.dynamic_obs.items():
            region, cond = val
            if cond and self.in_region(pos, region):
                flag = False
                break

        return flag

    def is_terminal(self):
        if self.in_region(self.curr_pos, self.G1):
            return 1
        elif self.steps_taken >= self.max_steps:
            return 1
        else:
            return 0

    def in_region(self, pos, region):
        x0, y0 = pos
        x1, y1, x2, y2 = region
        if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
            return True
        else:
            return False


if __name__=="__main__":
    # Random Agent
    import cv2
    rewards_list = []
    env = Gridworld_CL(debug=True, n_actions=8, change_interval=-1, change_count=3, max_steps=100,simplify_action=True,image_output=False)
    # print(env.motions)
    # plt.figure(3)
    # normalize_motion = (env.motions - np.min(env.motions)) / (np.max(env.motions) - np.min(env.motions))
    # RGB_list = np.column_stack((normalize_motion,0.5 * np.ones([len(env.motions),1])))
    # x = env.motions[:, 0]
    # y = env.motions[:, 1]
    # fig, ax = plt.subplots()
    # ax.scatter(x, y, c=RGB_list)
    # plt.xlim(-1,1)
    # delta = 0
    # unit = 0.01
    # for i, txt in enumerate(range(len(env.motions))):
    #     ax.annotate(txt, (x[i]+delta, y[i]))
    #     delta+=unit
    # plt.show()
    for i in range(1000):
        rewards = 0
        done = False
        state = env.reset()
        while not done:
            env.render_original()
            action =  env.action_space.sample()
            # print("action:"+str(action))
            next_state, r, done, info = env.step(action)
            rewards += r
            print(next_state)
            # pic = cv2.resize(next_state, (1000, 1000))
            # cv2.imshow("1", pic)
            # cv2.waitKey(50)
            if done:
                print(rewards)
            # time.sleep(2)
        # print(env.heatmap)
        # plt.figure(2)
        # plt.cla()
        # plt.imshow(env.heatmap[::-1])
        # plt.show()
        rewards_list.append(rewards)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))