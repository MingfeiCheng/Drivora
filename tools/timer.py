#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides access to the CARLA game time and contains a py_trees
timeout behavior using the CARLA game time
"""

import datetime

class GameTime(object):

    """
    This (static) class provides access to the CARLA game time.

    The elapsed game time can be simply retrieved by calling:
    GameTime.get_time()
    """

    _current_game_time = 0.0  # Elapsed game time after starting this Timer
    _carla_time = 0.0
    _last_frame = 0
    _platform_timestamp = 0
    _init = False

    @staticmethod
    def on_carla_tick(timestamp):
        """
        Callback receiving the CARLA time
        Update time only when frame is more recent that last frame
        """
        if GameTime._last_frame < timestamp.frame:
            frames = timestamp.frame - GameTime._last_frame if GameTime._init else 1
            GameTime._current_game_time += timestamp.delta_seconds * frames
            GameTime._last_frame = timestamp.frame
            GameTime._platform_timestamp = datetime.datetime.now()
            GameTime._init = True
            GameTime._carla_time = timestamp.elapsed_seconds

    @staticmethod
    def restart():
        """
        Reset game timer to 0
        """
        GameTime._current_game_time = 0.0
        GameTime._init = False

    @staticmethod
    def restart_with_server():
        GameTime._current_game_time = 0.0  # Elapsed game time after starting this Timer
        GameTime._carla_time = 0.0
        GameTime._last_frame = 0
        GameTime._platform_timestamp = 0
        GameTime._init = False

    @staticmethod
    def get_time():
        """
        Returns elapsed game time
        """
        return GameTime._current_game_time

    @staticmethod
    def get_carla_time():
        """
        Returns elapsed game time
        """
        return GameTime._carla_time

    @staticmethod
    def get_wallclocktime():
        """
        Returns elapsed game time
        """
        return GameTime._platform_timestamp

    @staticmethod
    def get_frame():
        """
        Returns elapsed game time
        """
        return GameTime._last_frame
