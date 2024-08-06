#!/usr/bin/env python3

from __future__ import annotations

import gym

from .envs.blockedunlockpickup import BlockedUnlockPickup
from .envs.unlockpickup import UnlockPickup
from .roomgrid import RoomGrid, reject_next_to
from .window import Window
from .wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from gym import spaces
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, Key, Ball, Box
from gym.core import ObservationWrapper, Wrapper
import numpy as np
import math

class unlockpickupUncertainGoalsactionbouns(UnlockPickup):
    def __init__(self, tile_size=8, seeds=[0], seed_idx=0, goal=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        self.tile_size = tile_size
        self.goal = goal
        self.counts = {}
        super().__init__()
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height * tile_size, self.width * tile_size, 3),
            dtype='uint8'
        )

    def _gen_grid(self, width, height):
        RoomGrid._gen_grid(self,width, height)

        # Add a box to the room on the right
        obj, pos1 = self.add_object(1, 0, kind="box")
        self.obj = obj

        obj2, pos2 = self.add_object_fixedpos(1, 0, posx=9, posy=4, kind="ball", color="purple")
        self.obj2 = obj2
        self.obj_list = [self.obj,self.obj2]

        # for debug
        # print("pos1:"+str(pos1))
        # print("pos2:"+str(pos2))

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)
        self.place_agent(0, 0)

        if self.goal == 0:
            self.real_goal = self.obj
            self.mission = "pick up the %s %s" % (self.obj.color, self.obj.type)
        else:
            self.real_goal = self.obj2
            self.mission = "pick up the %s %s" % (self.obj2.color, self.obj2.type)

    def add_object_fixedpos(self, i, j, posx, posy,kind=None, color=None):
        """
        Add a new object to room (i, j)
        """

        if kind == None:
            kind = self._rand_elem(['key', 'ball', 'box'])

        if color == None:
            color = self._rand_color()

        # TODO: we probably want to add an Object.make helper function
        assert kind in ['key', 'ball', 'box']
        if kind == 'key':
            obj = Key(color)
        elif kind == 'ball':
            obj = Ball(color)
        elif kind == 'box':
            obj = Box(color)

        return self.place_in_room_fixedpos(i, j, posx, posy, obj)

    def place_in_room_fixedpos(self, i, j, posx, posy, obj):
        """
        Add an existing object to room (i, j)
        """

        room = self.get_room(i, j)

        pos = self.place_obj_fixedpos(
            obj,
            posx,
            posy,
            room.top,
            room.size,
            reject_fn=reject_next_to,
            max_tries=1000
        )

        room.objs.append(obj)

        return obj, pos

    def place_obj_fixedpos(self,
        obj,
        posx,
        posy,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        pos_initial = np.array((
                posx,
                posy
            ))
        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            if num_tries == 1:
                pos = pos_initial

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos


    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        obs, reward, done, info = RoomGrid.step(self,action)

        # if action == self.actions.pickup:
        #     if self.carrying and self.carrying == self.real_goal:
        #         reward = self._reward()
        #         done = True

        env = self.unwrapped
        # tup = (tuple(env.agent_pos), env.agent_dir, action)
        tup = (tuple(env.agent_pos))
        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)

        # note: reward is only action bonus
        reward = bonus
        info["counts"] = self.counts

        return self.observation(obs), reward, done, info

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return rgb_img

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        seed = self.seeds[self.seed_idx]
        super().seed(seed)
        obs = super().reset(**kwargs)
        return self.observation(obs)

if __name__ == "__main__":
    env = unlockpickupUncertainGoals()
    env.reset()
    print("okk")