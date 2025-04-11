"""
Run from \szb_commons\commons_codebase\:
python -m pytest .\tests\test_core.py

Use decorator to skip a test temporarily:
@unittest.skip('wip')
"""

import sys
path_szb_commons = 'C:/Users/szb37/My Drive/Work efforts/szb_commons/'
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


class GetDfMeasureTests(DataWranglTests):

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


class GetDfVitalsTests(DataWranglTests):

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


class CalcScoresTests(DataWranglTests):

    def test_case0_calc_scores(self):
        ''' Case of not summing anything '''

        df, df_missingitems = core.DataWrangl.calc_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            measure_param = {
                'col_items': [],
                'col_score': 'tadaa',
                'col_complete': 'ebi_complete'})

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_calc_scores_case0.csv'))
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_case1_calc_scores(self):
        ''' Intended case '''

        # Calculate
        df, df_missingitems = core.DataWrangl.calc_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            measure_param = {
                'col_items': ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
                'col_score': 'tadaa',
                'col_complete': 'ebi_complete',})

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_calc_scores_case1.csv'))
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_case2_calc_scores(self):
        ''' Test normalization '''

        # Calculate
        df, df_missingitems = core.DataWrangl.calc_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            measure_param = {
                'col_items': ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
                'col_score': 'tadaa',
                'col_complete': 'ebi_complete',})

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_calc_scores_case2.csv'))
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_nitems case
        # Calculate
        df, df_missingitems = core.DataWrangl.calc_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            measure_param = {
                'col_items': ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
                'col_score': 'tadaa',
                'col_complete': 'ebi_complete',
                'norm_factor': 6,})
                #'normalize': 'by_nitems',})

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_calc_scores_case21.csv'))
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_maxscore case
        # Calculate
        df, df_missingitems = core.DataWrangl.calc_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            measure_param = {
                'col_items': ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
                'col_score': 'tadaa',
                'col_complete': 'ebi_complete',
                'norm_factor': 6*5,})
                #'normalize': 'by_maxscore',
                #'max_item_score': 5,},)

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_calc_scores_case22.csv'))
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

        ### Normalize by_maxscore case; change max value
        # Calculate
        df, df_missingitems = core.DataWrangl.calc_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            measure_param = {
                'col_items': ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
                'col_score': 'tadaa',
                'col_complete': 'ebi_complete',
                'norm_factor': 6*2,})
                #'normalize': 'by_maxscore',
                #'max_item_score': 2,})

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_calc_scores_case23.csv'))
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    def test_case3_calc_scores(self):
        ''' Missing data case. Should also produce print (use "python -m pytest -s .\tests\" to see in terminal):
                "Missing items from sum score calculation at row index (will skip rows from sum scores): [0, 1]"
        '''

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv'))
        df_redcap.iloc[0, 2] = math.nan
        df_redcap.iloc[1, 2] = None
        df_redcap.iloc[1, 3] = None

        # Calculate
        df, df_missingitems = core.DataWrangl.calc_scores(
            df_redcap = df_redcap,
            measure_param = {
                'measure': 'mock_measure',
                'col_items': ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6',],
                'col_score': 'tadaa',
                'col_complete': 'ebi_complete',})

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_calc_scores_case3.csv'))
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert df_solution.equals(df)

    ''' Test revserse scoring '''
    def test_case4_calc_scores(self):

        measure_param = {
            'instrument': 'mock',
            'measure_type': 'change',
            'col_complete': 'complete',
            'measure': 'mock',
            'col_score': 'mock_score_local',
            'col_items': [f'mock_q{idx}' for idx in range(1,6)],
            'reverse_items': ['mock_q1', 'mock_q3', 'mock_q5',],
            'min_score': 0,
            'max_score': 100,}

        df = pd.DataFrame({
            'pID': [1, 2, 3, 4],
            'tp': ['bsl']*4,
            'complete': [2]*4,
            'mock_q1': [0  , 100, 30, 10,],
            'mock_q2': [100,   0, 70, 10,],
            'mock_q3': [0  , 100, 25, 10,],
            'mock_q4': [100,   0, 50, 10,],
            'mock_q5': [0  , 100, 90, 10,],
        })

        df_solution = pd.DataFrame({
            'pID': [1, 2, 3, 4],
            'tp': ['bsl']*4,
            'complete': [2]*4,
            'mock_q1': [0  , 100, 30, 10,],
            'mock_q2': [100,   0, 70, 10,],
            'mock_q3': [0  , 100, 25, 10,],
            'mock_q4': [100,   0, 50, 10,],
            'mock_q5': [0  , 100, 90, 10,],
             'mock_score_local': [500.0, 0, (70+70+75+50+10), (90*3+10*2)]
        })

        # Calculate & compare to solution
        df, df_missingitems = core.DataWrangl.calc_scores(df, measure_param)
        assert df_solution.equals(df)

    def test_case5_calc_scores(self):

        measure_param = {
            'instrument': 'mock',
            'measure_type': 'change',
            'col_complete': 'complete',
            'measure': 'mock',
            'col_score': 'mock_score_local',
            'col_items': [f'mock_q{idx}' for idx in range(1,6)],
            'reverse_items': ['mock_q1', 'mock_q2'],
            'min_score': 1,
            'max_score': 5,}

        df = pd.DataFrame({
            'pID': [1, 2, 3, 4],
            'tp': ['bsl']*4,
            'complete': [2]*4,
            'mock_q1': [1 , 5, 3, 5,],
            'mock_q2': [1,  5, 3, 2,],
            'mock_q3': [5 , 1, 2, 3,],
            'mock_q4': [5,  1, 5, 1,],
            'mock_q5': [5 , 1, 2, 1,],
        })

        df_solution = pd.DataFrame({
            'pID': [1, 2, 3, 4],
            'tp': ['bsl']*4,
            'complete': [2]*4,
            'mock_q1': [1 , 5, 3, 5,],
            'mock_q2': [1,  5, 3, 2,],
            'mock_q3': [5 , 1, 2, 3,],
            'mock_q4': [5,  1, 5, 1,],
            'mock_q5': [5 , 1, 2, 1,],
            'mock_score_local': [25.0, 5, (3+3+2+5+2), (1+4+3+1+1)]
        })

        # Calculate & compare to solution
        df, df_missingitems = core.DataWrangl.calc_scores(df, measure_param)
        assert df_solution.equals(df)

    def test_case6_calc_scores(self):

        measure_param = {
            'instrument': 'mock',
            'measure_type': 'change',
            'col_complete': 'complete',
            'measure': 'mock',
            'col_score': 'mock_score_local',
            'col_items': [f'mock_q{idx}' for idx in range(1,6)],
            'reverse_items': ['mock_q1', 'mock_q2'],
            'min_score': 0,
            'max_score': 5,}

        df = pd.DataFrame({
            'pID': [1, 2, 3, 4],
            'tp': ['bsl']*4,
            'complete': [2]*4,
            'mock_q1': [0 , 5, 3, 5,],
            'mock_q2': [0,  5, 0, 2,],
            'mock_q3': [5 , 0, 2, 3,],
            'mock_q4': [5,  0, 5, 0,],
            'mock_q5': [5 , 0, 2, 1,],
        })

        df_solution = pd.DataFrame({
            'pID': [1, 2, 3, 4],
            'tp': ['bsl']*4,
            'complete': [2]*4,
            'mock_q1': [0 , 5, 3, 5,],
            'mock_q2': [0,  5, 0, 2,],
            'mock_q3': [5 , 0, 2, 3,],
            'mock_q4': [5,  0, 5, 0,],
            'mock_q5': [5 , 0, 2, 1,],
            'mock_score_local': [25.0, 0, (2+5+2+5+2), (0+3+3+0+1)]
        })

        # Calculate & compare to solution
        df, df_missingitems = core.DataWrangl.calc_scores(df, measure_param)
        assert df_solution.equals(df)


class AddDeltaScoresTests(DataWranglTests):

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


class WidenMasterTests(DataWranglTests):

    def test_case0_widen_master(self):

        # Calculate
        df = core.DataWrangl.widen_master(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_v2.csv')),
            xvars= ['bsl1', 'bsl2'],
            x_tp = 'bsl' ,
            x_use_delta=False,
            yvars = ['measure1', 'measure2'],
            y_tp = 'A28',
            y_use_delta=True)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_widen_master_case0.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df_solution['bsl1'] = df_solution['bsl1'].astype('float64')
        df_solution['bsl2'] = df_solution['bsl2'].astype('float64')
        df_solution['measure1'] = df_solution['measure1'].astype('float64')
        df_solution['measure2'] = df_solution['measure2'].astype('float64')

        assert df_solution.equals(df)

    def test_case1_widen_master(self):

        # Calculate
        df = core.DataWrangl.widen_master(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_v2.csv')),
            xvars= ['bsl1',],
            x_tp = 'bsl' ,
            x_use_delta=False,
            yvars = ['measure2'],
            y_tp = 'B28',
            y_use_delta=False)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_widen_master_case1.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df_solution['bsl1'] = df_solution['bsl1'].astype('float64')
        df_solution['measure2'] = df_solution['measure2'].astype('float64')

        assert df_solution.equals(df)

    def test_case2_widen_master(self):

        # Calculate
        df = core.DataWrangl.widen_master(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_v2.csv')),
            xvars= ['bsl1', 'bsl2'],
            x_tp = 'bsl' ,
            x_use_delta=True,
            yvars = ['measure1', 'measure2'],
            y_tp = 'A28',
            y_use_delta=True)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_widen_master_case2.csv'))

        # Compare
        df_solution.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df_solution['bsl1'] = df_solution['bsl1'].astype('float64')
        df_solution['bsl2'] = df_solution['bsl2'].astype('float64')
        df_solution['measure1'] = df_solution['measure1'].astype('float64')
        df_solution['measure2'] = df_solution['measure2'].astype('float64')

        assert df_solution.equals(df)


class AnalysisTests(unittest.TestCase):
    pass


class GetDfObservedTests(AnalysisTests):

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


class GetDfTpNdaysTests(AnalysisTests):

    def test_case0_get_df_tp_ndays(self):

        # Calculate
        df = core.DataWrangl.get_df_tp_ndays(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_wdates_case0.csv')),
            col_date = 'date')
        df.reset_index(drop=True, inplace=True)

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_ get_df_tp_ndays_case0.csv'))
        assert df_solution.equals(df)

    def test_case1_get_df_tp_ndays(self):

        # Calculate
        df = core.DataWrangl.get_df_tp_ndays(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_wdates_case1.csv')),
            col_date = 'date')
        df.reset_index(drop=True, inplace=True)

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_ get_df_tp_ndays_case1.csv'))
        assert df_solution.equals(df)


class GetCorrmatTests(AnalysisTests):

    def test_case0_get_corrmat(self):

        df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_for_corrmat.csv'))
        vars_outcome = ['MADRS', 'PANSS', 'ISI', 'YMRS', 'BRQ', 'QOLBD', 'ECRM16_anx', 'ECRM16_avoid']
        vars_bsl = ['ACE', 'PCL5', 'TE_sum', 'TE_pos', 'TE_neg',]

        df = core.DataWrangl.widen_master(
            df_master = df_master,
            xvars = vars_bsl,
            x_tp = 'bsl',
            x_use_delta = False,
            yvars = vars_outcome,
            y_tp = 'A21',
            y_use_delta = True)

        ### Check Pearson correlation for A21-bsl_vs_baseline case
        df_coeffs, df_pvalues = core.Analysis.get_corrmats(
            df = df,
            xvars = vars_bsl,
            yvars = vars_outcome,
            methods = ['pearson',],
            draw = False)

        df_solution_coeffs = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_baseline]_pearson_coeffs.csv'), index_col=0)
        df_solution_pvalues = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_baseline]_pearson_pvalues.csv'), index_col=0)
        assert df_solution_coeffs.equals(df_coeffs)
        assert df_solution_pvalues.equals(df_pvalues)

        ### Check Spearman correlation for A21-bsl_vs_baseline case
        df_coeffs, df_pvalues = core.Analysis.get_corrmats(
            df = df,
            xvars = vars_bsl,
            yvars = vars_outcome,
            draw = False,
            methods = ['spearman',],)
        df_solution_coeffs = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_baseline]_spearman_coeffs.csv'), index_col=0)
        df_solution_pvalues = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_baseline]_spearman_pvalues.csv'), index_col=0)
        assert df_solution_coeffs.equals(df_coeffs)
        assert df_solution_pvalues.equals(df_pvalues)

        ### Check Kendall correlation for A21-bsl_vs_baseline case
        df_coeffs, df_pvalues = core.Analysis.get_corrmats(
            df = df,
            xvars = vars_bsl,
            yvars = vars_outcome,
            draw = False,
            methods = ['kendall',],)
        df_solution_coeffs = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_baseline]_kendall_coeffs.csv'), index_col=0)
        df_solution_pvalues = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_baseline]_kendall_pvalues.csv'), index_col=0)
        assert df_solution_coeffs.equals(df_coeffs)
        assert df_solution_pvalues.equals(df_pvalues)

    def test_case1_get_corrmat(self):

        df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_for_corrmat.csv'))
        vars_outcome = ['MADRS', 'PANSS', 'ISI', 'YMRS', 'BRQ', 'QOLBD', 'ECRM16_anx', 'ECRM16_avoid']
        vars_session = ['EBI', 'PIQ', 'TEQ', 'CEQ', 'MEQ_sum', 'MEQ_myst', 'MEQ_mood', 'MEQ_tran', 'MEQ_inef']

        df = core.DataWrangl.widen_master(
            df_master = df_master,
            xvars = vars_session,
            x_tp = 'A0',
            x_use_delta = False,
            yvars = vars_outcome,
            y_tp = 'A21',
            y_use_delta = True)

        ### Check Pearson correlation for A21-bsl_vs_baseline case
        df_coeffs, df_pvalues = core.Analysis.get_corrmats(
            df = df,
            xvars = vars_session,
            yvars = vars_outcome,
            draw = False,
            methods = ['pearson',],)
        df_solution_coeffs = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_A0]_pearson_coeffs.csv'), index_col=0)
        df_solution_pvalues = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_A0]_pearson_pvalues.csv'), index_col=0)
        assert df_solution_coeffs.equals(df_coeffs)
        assert df_solution_pvalues.equals(df_pvalues)

        ### Check Spearman correlation for A21-bsl_vs_baseline case
        df_coeffs, df_pvalues = core.Analysis.get_corrmats(
            df = df,
            xvars = vars_session,
            yvars = vars_outcome,
            draw = False,
            methods = ['spearman',],)
        df_solution_coeffs = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_A0]_spearman_coeffs.csv'), index_col=0)
        df_solution_pvalues = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_A0]_spearman_pvalues.csv'), index_col=0)
        assert df_solution_coeffs.equals(df_coeffs)
        assert df_solution_pvalues.equals(df_pvalues)

        ### Check Kendall correlation for A21-bsl_vs_baseline case
        df_coeffs, df_pvalues = core.Analysis.get_corrmats(
            df = df,
            xvars = vars_session,
            yvars = vars_outcome,
            draw = False,
            methods = ['kendall',],)
        df_solution_coeffs = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_A0]_kendall_coeffs.csv'), index_col=0)
        df_solution_pvalues = pd.read_csv(os.path.join(folders.fixtures_out, 'corrmat_[A21-bsl_vs_A0]_kendall_pvalues.csv'), index_col=0)
        assert df_solution_coeffs.equals(df_coeffs)
        assert df_solution_pvalues.equals(df_pvalues)


class InsertHAMDequalScore(AnalysisTests):

    def test_convert_toHAMD(self):

        assert core.Helpers.convert_toHAMD(14, 'BDI', delta=False)==(12.5, None)
        assert core.Helpers.convert_toHAMD(14, 'BDI1', delta=False)==(12.5, None)
        assert core.Helpers.convert_toHAMD(9, 'BDI', delta=False)==(9, None)
        assert core.Helpers.convert_toHAMD(14, 'BDI', delta=True)==(9.5, None)
        assert core.Helpers.convert_toHAMD(-14, 'BDI', delta=True)==(-9.5, None)
        assert core.Helpers.convert_toHAMD(-1, 'BDI', delta=False)== \
            (math.nan, 'Some scores are below the defined minimum and cannot be converted to HAMD17; converts to math.nan')
        assert core.Helpers.convert_toHAMD(200, 'BDI', delta=False)== \
            (math.nan, 'Some scores are above the defined minimum and cannot be converted to HAMD17; converts to math.nan')

        assert core.Helpers.convert_toHAMD(4, 'MADRS', delta=False)==(4, None)
        assert core.Helpers.convert_toHAMD(29, 'MADRS', delta=False)==(23, None)
        assert core.Helpers.convert_toHAMD(30, 'MADRS', delta=False)==(23, None)
        assert core.Helpers.convert_toHAMD(33.2, 'MADRS', delta=False)==(25.2, None)
        assert core.Helpers.convert_toHAMD(33.7, 'MADRS', delta=False)==(25.7, None)
        assert core.Helpers.convert_toHAMD(14, 'MADRS', delta=True)==(12, None)
        assert core.Helpers.convert_toHAMD(-14, 'MADRS', delta=True)==(-12, None)
        assert core.Helpers.convert_toHAMD(-1, 'MADRS', delta=False)== \
            (math.nan, 'Some scores are below the defined minimum and cannot be converted to HAMD17; converts to math.nan')
        assert core.Helpers.convert_toHAMD(200, 'MADRS', delta=False)== \
            (math.nan, 'Some scores are above the defined minimum and cannot be converted to HAMD17; converts to math.nan')

        assert core.Helpers.convert_toHAMD(0, 'AGYFASZ', delta=False)== \
            (math.nan, f'HAMD17 eq scores are not defined for scale AGYFASZ; converts to math.nan')
        assert core.Helpers.convert_toHAMD(0, 'AGYFASZ', delta=True)== \
            (math.nan, f'HAMD17 eq Δ scores are not defined for scale AGYFASZ; converts to math.nan')

    def test_case0_insert_HAMDequal_score(self):

        df = pd.read_csv(os.path.join(folders.fixtures_in, 'df_hamdconvert_v0.csv'))

        df = core.Analysis.insert_HAMDequal_score(df, cols=['mean'], delta=False)
        df = core.Analysis.insert_HAMDequal_score(df, cols=['delta_mean'], delta=True)

        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_insertHAMDeq_case0.csv'))
        assert df_solution.equals(df)
