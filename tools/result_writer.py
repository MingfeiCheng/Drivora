#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains the result gatherer and write for CARLA scenarios.
It shall be used from the ScenarioManager only.
"""

from __future__ import print_function

import time

from tabulate import tabulate

class ResultOutputProvider(object):

    """
    This module contains the _result gatherer and write for CARLA scenarios.
    It shall be used from the ScenarioManager only.
    """

    def __init__(self, data, result, stdout=True, filename=None, junit=None):
        """
        Setup all parameters
        - _data contains all scenario-related information
        - _result is overall pass/fail info
        - _stdout (True/False) is used to (de)activate terminal output
        - _filename is used to (de)activate file output in tabular form
        - _junit is used to (de)activate file output in _junit form
        """
        self._data = data
        self._result = result
        self._stdout = stdout
        self._filename = filename
        self._junit = junit

        self._start_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                         time.localtime(self._data.start_system_time))
        self._end_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(self._data.end_system_time))

    def write(self):
        """
        Public write function
        """
        output = self.create_output_text()
        if self._filename is not None:
            with open(self._filename, 'w', encoding='utf-8', errors='replace') as fd:
                fd.write(output)
        if self._stdout:
            print(output)

    def create_output_text(self):
        """
        Creates the output message
        """
        output = "\n"
        output += " ======= Results of Scenario: {} ---- {} =======\n".format(
            self._data.scenario_instance.name, self._result)
        end_line_length = len(output) - 3
        output += "\n"

        # Lis of all the actors
        output += " > Ego vehicles:\n"
        for ego_vehicle in self._data.scenario_instance.ego_vehicles:
            output += "{}; ".format(ego_vehicle)
        output += "\n\n"

        output += " > Other actors:\n"
        for actor in self._data.scenario_instance.other_actors:
            output += "{}; ".format(actor)
        output += "\n\n"

        # Simulation part
        output += " > Simulation Information\n"

        system_time = round(self._data.scenario_duration_system, 2)
        game_time = round(self._data.scenario_duration_game, 2)
        ratio = round(self._data.scenario_duration_game / self._data.scenario_duration_system, 3)

        list_statistics = [["Start Time", "{}".format(self._start_time)]]
        list_statistics.extend([["End Time", "{}".format(self._end_time)]])
        list_statistics.extend([["Duration (System Time)", "{}s".format(system_time)]])
        list_statistics.extend([["Duration (Game Time)", "{}s".format(game_time)]])
        list_statistics.extend([["Ratio (System Time / Game Time)", "{}s".format(ratio)]])

        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n\n"

        # Criteria part
        output += " > Criteria Information\n"
        header = ['Actor', 'Criterion', 'Result', 'Actual Value', 'Success Value']
        list_statistics = [header]

        for criterion in self._data.scenario_instance.get_criteria():
            name = criterion.name
            if criterion.optional:
                name += " (Opt.)"
            else:
                name += " (Req.)"

            try:
                actor = "{} (id={})".format(criterion.actor.type_id[8:], criterion.actor.id)
            except Exception:
                actor = "GroupTest (id=GroupTest)"
            
            list_statistics.extend([[
                actor, name, criterion.test_status, criterion.actual_value, criterion.success_value]])

        # Timeout
        actor = ""
        criteria = "Timeout (Req.)"
        result = "SUCCESS" if self._data.scenario_duration_game < self._data.scenario_instance.timeout else "FAILURE"
        actual_value = round(self._data.scenario_duration_game, 2)
        expected_value = round(self._data.scenario_instance.timeout, 2)

        list_statistics.extend([[actor, criteria, result, actual_value, expected_value]])

        # Global and final output message
        list_statistics.extend([['', 'GLOBAL RESULT', self._result, '', '']])

        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n"
        output += " " + "=" * end_line_length + "\n"

        return output
