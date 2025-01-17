"""
Run from \szb_commons\commons_codebase\:
    python -m pytest .\tests\

Use decorator to skip a test temporarily:
    @unittest.skip('wip')
"""

import sys
path_szb_commons = 'C://Users//szb37//My Drive//Efforts//szb_commons'
sys.path.append(path_szb_commons)

import src.core as core
import src.folders as folders
import src.config as config
from unittest import mock
import pandas as pd
import unittest
import pytest
import math
import os


class DataWranglTests(unittest.TestCase):
    pass

class get_df_measure_Tests(DataWranglTests):

    ''' Testing measures without time '''
    def test_case0_get_df_measure(self):
        ''' Case of missing scores '''

        # Calculate
        df = core.DataWrangl.get_df_measure(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v0.csv')),
            measure_param = {
                'instrument': 'EBI',
                'measure_type': 'post_dose',
                'measure': 'EBI',
                'col_complete': None,
                'col_score': 'ebi_score',})

        # Get manual solution (sort types) and compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_get_df_measure_case0.csv'))
        df_solution['pID'] = df_solution['pID'].astype('int64')
        df_solution['time'] = df_solution['time'].astype('float64')
        df_solution['score'] = df_solution['score'].astype('float64')
        assert df_solution.equals(df)

    def test_case1_get_df_measure(self):
        ''' Intended use case '''

        # Calculate
        df = core.DataWrangl.get_df_measure(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            measure_param = {
                'instrument': 'EBI',
                'measure_type': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out,
            'df_solution_get_df_measure_case1.csv'))
        assert df_solution.equals(df)

    def test_case2_get_df_measure(self):
        ''' Check if col_complete=None works as intended '''

        # Calculate
        df = core.DataWrangl.get_df_measure(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            measure_param = {
                'instrument': 'EBI',
                'measure_type': 'post_dose',
                'measure': 'EBI',
                'col_complete': None,
                'col_score': 'ebi_score',})

        # Get manual solution & solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out,
            'df_solution_get_df_measure_case2.csv'))
        assert df_solution.equals(df)

    def test_case3_get_df_measure(self):
        ''' Check if None and math.nan scores removed '''

        # Edit scores
        df_redcap1 = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv'))
        df_redcap1.iloc[0, 8] = math.nan
        df_redcap1.iloc[1, 8] = None

        # Calculate
        df = core.DataWrangl.get_df_measure(
            df_redcap = df_redcap1,
            measure_param = {
                'instrument': 'EBI',
                'measure_type': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out,
            'df_solution_get_df_measure_case3.csv'))
        assert df_solution.equals(df)

    def test_case4_get_df_measure(self):
        ''' Check whether extending cols_to_keep works '''

        # Calculate
        df = core.DataWrangl.get_df_measure(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            cols_to_keep = ['pID', 'tp', 'ebi_1', 'ebi_2'],
            measure_param = {
                'instrument': 'EBI',
                'measure_type': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out,
            'df_solution_get_df_measure_case4.csv'))
        assert df_solution.equals(df)

    ''' Testing measures with time '''
    def test_case5_get_df_measure(self):

        # Calculate
        df = core.DataWrangl.get_df_measure(
            pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_indosemeasures_v0.csv')),
            config.indose_measure_params[0],)

        # Get manual solution and compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_get_df_measure_case5.csv'))
        assert df_solution.equals(df)

    def test_case6_get_df_measure(self):

        # Calculate
        df = core.DataWrangl.get_df_measure(
            pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_indosemeasures_v0.csv')),
            config.indose_measure_params[13],)

        # Get manual solution and compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_get_df_measure_case6.csv'))
        assert df_solution.equals(df)

    ''' Testing missing data case '''
    def test_case7_get_df_measure(self):

        # Calculate
        df = core.DataWrangl.get_df_measure(
            pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_indosemeasures_v1.csv')),
            config.indose_measure_params[5],)

        # Get manual solution and compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_get_df_measure_case7.csv'))
        assert df_solution.equals(df)


class get_df_vitals_Tests(DataWranglTests):

    def test_case0_get_df_vitals(self):

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_vitalsonly_case0.csv'))
        df_redcap = core.DataWrangl.format_bsl_vitals(df_redcap)
        df = core.DataWrangl.get_df_vitals(df_redcap, config.vitals_measure_params[0],)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out,
            'df_solution_get_df_vitals_case0.csv'))
        assert df_solution.equals(df)

    def test_case1_get_df_vitals(self):

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_vitalsonly_case0.csv'))
        df_redcap = core.DataWrangl.format_bsl_vitals(df_redcap)
        df = core.DataWrangl.get_df_vitals(df_redcap, config.vitals_measure_params[17],)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out,
            'df_solution_get_df_vitals_case1.csv'))
        assert df_solution.equals(df)

    ''' Testing missing data case '''
    def test_case2_get_df_vitals(self):

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_vitalsonly_case1.csv'))
        df_redcap = core.DataWrangl.format_bsl_vitals(df_redcap)
        df = core.DataWrangl.get_df_vitals(df_redcap, config.vitals_measure_params[17],)
        df['pID'] = df['pID'].astype('int64') # due to missing data pIDs are read as floats, converting type

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out,
            'df_solution_get_df_vitals_case2.csv'))
        assert df_solution.equals(df)

class add_sum_scores_Tests(DataWranglTests):

    def test_case0_add_sum_scores(self):
        ''' Case of not summing anything '''

        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            col_items = [],
            col_score = 'tadaa',
            col_complete = 'ebi_complete')

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_sum_scores_case0.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_case1_add_sum_scores(self):
        ''' Intended case '''

        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete')

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_sum_scores_case1.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_case2_add_sum_scores(self):
        ''' Test normalization '''

        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete')

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_sum_scores_case2.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_nitems case
        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete',
            normalize = 'by_nitems',)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_sum_scores_case21.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_maxscore case
        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete',
            normalize = 'by_maxscore',
            max_item_score = 5,)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_sum_scores_case22.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_maxscore case; change max value
        # Calculate
        df = core.DataWrangl.add_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            col_items = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
            col_score = 'tadaa',
            col_complete = 'ebi_complete',
            normalize = 'by_maxscore',
            max_item_score = 2,)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_sum_scores_case23.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_case3_add_sum_scores(self):
        ''' Missing data case. Should also produce print (use "python -m pytest -s .\tests\" to see in terminal):
                "Missing items from sum score calculation at row index (will skip rows from sum scores): [0, 1]"
        '''

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv'))
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
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_sum_scores_case3.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

class add_delta_scores_Tests(DataWranglTests):

    def test_case1_add_delta_scores(self):
        ''' Intended use case '''

        # Calculate
        df = core.DataWrangl.add_delta_scores(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_with_time_v0.csv')),
            delta_from_tp = 'bsl',
            delta_from_time = 15)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_delta_scores_case1.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ''' Change both delta_from inuts '''

        # Calculate
        df = core.DataWrangl.add_delta_scores(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_with_time_v0.csv')),
            delta_from_tp = 'A28',
            delta_from_time = 30)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_delta_scores_case11.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_case2_add_delta_scores(self):
        ''' Testing case of undecided_has_time
            Should also print message "Can not decide whether measure has time: ['EBI', 'INT_fake']"
        '''

        # Calculate
        df = core.DataWrangl.add_delta_scores(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_with_time_v1.csv')),
            delta_from_tp = 'bsl',
            delta_from_time = 15)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_add_delta_scores_case2.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)


class AnalysisTests(unittest.TestCase):
    pass

class get_df_observed_Tests(AnalysisTests):

    def test_case0_get_df_observed(self):

        # Calculate
        df = core.Analysis.get_df_observed(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_v0.csv')),)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_get_df_observed_case0.csv'))

        # Compare
        assert df_solution.equals(df)

    def test_case1_get_df_observed(self):

        # Calculate
        df = core.Analysis.get_df_observed(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_v1.csv')),)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_get_df_observed_case1.csv'))

        # Compare
        assert df_solution.equals(df)

class get_df_tp_ndays_Tests(AnalysisTests):

    def test_case0_get_df_tp_ndays(self):

        # Calculate
        df = core.Analysis.get_df_tp_ndays(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_wdates_case0.csv')),)
        df.reset_index(drop=True, inplace=True)

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_ get_df_tp_ndays_case0.csv'))
        assert df_solution.equals(df)

    def test_case1_get_df_tp_ndays(self):

        # Calculate
        df = core.Analysis.get_df_tp_ndays(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_wdates_case1.csv')),)
        df.reset_index(drop=True, inplace=True)

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_ get_df_tp_ndays_case1.csv'))
        assert df_solution.equals(df)
