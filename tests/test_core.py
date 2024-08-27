"""
Run from codebase directory with
python -m pytest .\tests\
@unittest.skip('wip')
"""

import src.core as core
from unittest import mock
import pandas as pd
import unittest
import pytest
import math
import os

dir_inputs = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'fixtures',
    'inputs')
dir_outputs = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'fixtures',
    'expected_outputs')


class DataWranglTests(unittest.TestCase):

    def test_get_longdf_of_measure_case0(self):
        ''' Case of missing scores '''

        # Calculate
        df = core.DataWrangl.get_longdf_of_measure(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v0.csv')),
            measure = {
                'instrument': 'EBI',
                'type': 'post_dose',
                'measure': 'EBI',
                'col_complete': None,
                'col_score': 'ebi_score',})

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_get_longdf_of_measure_case0.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df_solution['pID'] = df_solution['pID'].astype('int64')
        df_solution['score'] = df_solution['score'].astype('float64')
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_get_longdf_of_measure_case1(self):
        ''' Intended use case '''

        # Calculate
        df = core.DataWrangl.get_longdf_of_measure(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v1.csv')),
            measure = {
                'instrument': 'EBI',
                'type': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs,
            'df_solution_get_longdf_of_measure_case1.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_get_longdf_of_measure_case2(self):
        ''' Check if col_complete=None works as intended '''

        # Calculate
        df = core.DataWrangl.get_longdf_of_measure(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v1.csv')),
            measure = {
                'instrument': 'EBI',
                'type': 'post_dose',
                'measure': 'EBI',
                'col_complete': None,
                'col_score': 'ebi_score',})

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs,
            'df_solution_get_longdf_of_measure_case2.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_get_longdf_of_measure_case3(self):
        ''' Check if None and math.nan scores removed '''

        # Edit scores
        df_redcap1 = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v1.csv'))
        df_redcap1.iloc[0, 8] = math.nan
        df_redcap1.iloc[1, 8] = None

        # Calculate
        df = core.DataWrangl.get_longdf_of_measure(
            df_redcap = df_redcap1,
            measure = {
                'instrument': 'EBI',
                'type': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs,
            'df_solution_get_longdf_of_measure_case3.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_get_longdf_of_measure_case4(self):
        ''' Check whether extending cols_to_keep works '''

        # Calculate
        df = core.DataWrangl.get_longdf_of_measure(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v1.csv')),
            cols_to_keep = ['pID', 'tp', 'ebi_1', 'ebi_2'],
            measure = {
                'instrument': 'EBI',
                'type': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs,
            'df_solution_get_longdf_of_measure_case4.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Case where cols_to_keep is empty

        # Calculate
        df = core.DataWrangl.get_longdf_of_measure(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v1.csv')),
            cols_to_keep = [],
            measure = {
                'instrument': 'EBI',
                'type': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs,
            'df_solution_get_longdf_of_measure_case41.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_add_sum_scores_case0(self):
        ''' Case of not summing anything '''

        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v1.csv')),
            col_items = [],
            col_score = 'tadaa',
            col_complete = 'ebi_complete')

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_sum_scores_case0.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_add_sum_scores_case1(self):
        ''' Intended case '''

        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v1.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete')

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_sum_scores_case1.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_add_sum_scores_case2(self):
        ''' Test normalization '''

        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v2.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete')

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_sum_scores_case2.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_item_number case
        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v2.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete',
            normalize = 'by_item_number',)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_sum_scores_case21.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_max_value case
        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v2.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete',
            normalize = 'by_max_value',
            max_value = 5,)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_sum_scores_case22.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_max_value case; change max value
        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v2.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete',
            normalize = 'by_max_value',
            max_value = 2,)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_sum_scores_case23.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_add_sum_scores_case3(self):
        ''' Missing data case. Should also produce print (use "python -m pytest -s .\tests\" to see in terminal):
                'add_sum_scores' sums incomplete columns at row index: 0
                'add_sum_scores' sums incomplete columns at row index: 1
        '''

        df_redcap = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_v1.csv'))
        df_redcap.iloc[0, 2] = math.nan
        df_redcap.iloc[1, 2] = None
        df_redcap.iloc[1, 3] = None

        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = df_redcap,
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete')

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_sum_scores_case3.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_add_delta_scores_case1(self):
        ''' Intended use case '''

        # Calculate
        df = core.DataWrangl.add_delta_scores(
            df_master = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_with_time_v0.csv')),
            delta_from_tp = 'bsl',
            delta_from_time = 15)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_delta_scores_case1.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ''' Change both delta_from inuts '''

        # Calculate
        df = core.DataWrangl.add_delta_scores(
            df_master = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_with_time_v0.csv')),
            delta_from_tp = 'A28',
            delta_from_time = 30)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_delta_scores_case11.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_add_delta_scores_case2(self):
        ''' Testing case of undecided_has_time
            Should also print message "Can not decide whether measure has time: ['EBI', 'INT_fake']"
        '''

        # Calculate
        df = core.DataWrangl.add_delta_scores(
            df_master = pd.read_csv(os.path.join(dir_inputs, 'df_redcap_with_time_v1.csv')),
            delta_from_tp = 'bsl',
            delta_from_time = 15)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(dir_outputs, 'df_solution_add_delta_scores_case2.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)
