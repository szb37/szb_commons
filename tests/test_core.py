"""
Tests of src/core.py. Most tests compare the output of a function to an expected output stored in
tests/fixtures/expected_outputs, computed from inputs in tests/fixtures/inputs.

Run from the repo root (imports are set up by tests/conftest.py):
    python -m pytest tests
"""

import commons_codebase.src.core as core
import commons_codebase.src.folders as folders
import commons_codebase.src.config as config
from unittest import mock
import pandas as pd
import unittest
import tempfile
import warnings
import math
import os
import re


def read_solution(fname):
    ''' Read an expected output fixture. Fixtures predate renaming the 'measure_type' column to 'timing'. '''
    df_solution = pd.read_csv(os.path.join(folders.fixtures_out, fname))
    return df_solution.rename(columns={'measure_type': 'timing'})


def to_current_vitals_format(df_redcap):
    ''' Vitals fixtures use an older REDCap naming: baseline readings are in vs_bl_{measure} and
        dosing readings in vs_dose{time}_{measure}{reading}. Rename them to the
        vs_dose{time}_{measure}_{reading} format expected by get_df_vitals(); baseline has a single reading.
    '''
    df_redcap = df_redcap.rename(columns=lambda col: re.sub(r'^(vs_dose\d+_(?:dia|sys|hr))([12])$', r'\1_\2', col))
    for measure in ['dia', 'sys', 'hr']:
        df_redcap = df_redcap.rename(columns={f'vs_bl_{measure}': f'vs_dose0_{measure}_1'})
        df_redcap[f'vs_dose0_{measure}_2'] = math.nan
    return df_redcap


def ebi_sum_score_param(col_items):
    ''' measure_param for calc_sum_scores() tests; the sum score is stored in column 'tadaa_LOCAL_SCORE' '''
    return {
        'measure': 'tadaa',
        'col_complete': 'ebi_complete',
        'sum_score': {
            'col_items': col_items,
            'reverse_items': [],
            'min_score': 0,
            'max_score': 100,},}


EBI_ITEMS = ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5', 'ebi_6']


class DataWranglTests(unittest.TestCase):
    pass


class GetDfMeasureTests(DataWranglTests):

    # Measures without time
    def test_case0_get_df_measure(self):
        ''' Case of missing scores '''

        # Calculate
        df = core.DataWrangl.get_df_measure(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v0.csv')),
            measure_param = {
                'instrument': 'EBI',
                'timing': 'post_dose',
                'measure': 'EBI',
                'col_complete': None,
                'col_score': 'ebi_score',})

        # Get manual solution (fix dtypes) and compare
        df_solution = read_solution('df_solution_get_df_measure_case0.csv')
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
                'timing': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution
        df_solution = read_solution('df_solution_get_df_measure_case1.csv')
        assert df_solution.equals(df)

    def test_case2_get_df_measure(self):
        ''' Check if col_complete=None works as intended '''

        # Calculate
        df = core.DataWrangl.get_df_measure(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            measure_param = {
                'instrument': 'EBI',
                'timing': 'post_dose',
                'measure': 'EBI',
                'col_complete': None,
                'col_score': 'ebi_score',})

        # Get manual solution & compare
        df_solution = read_solution('df_solution_get_df_measure_case2.csv')
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
                'timing': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution & compare
        df_solution = read_solution('df_solution_get_df_measure_case3.csv')
        assert df_solution.equals(df)

    def test_case4_get_df_measure(self):
        ''' Check whether extending cols_to_keep works '''

        # Calculate
        df = core.DataWrangl.get_df_measure(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            cols_to_keep = ['pID', 'tp', 'ebi_1', 'ebi_2'],
            measure_param = {
                'instrument': 'EBI',
                'timing': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'col_score': 'ebi_score',})

        # Get manual solution & compare
        df_solution = read_solution('df_solution_get_df_measure_case4.csv')
        assert df_solution.equals(df)

    # Measures with time
    def test_case5_get_df_measure(self):

        # Calculate
        df = core.DataWrangl.get_df_measure(
            pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_indosemeasures_v0.csv')),
            config.indose_measure_params[0],)

        # Get manual solution and compare
        df_solution = read_solution('df_solution_get_df_measure_case5.csv')
        assert df_solution.equals(df)

    def test_case6_get_df_measure(self):

        # Calculate
        df = core.DataWrangl.get_df_measure(
            pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_indosemeasures_v0.csv')),
            config.indose_measure_params[13],)

        # Get manual solution and compare
        df_solution = read_solution('df_solution_get_df_measure_case6.csv')
        assert df_solution.equals(df)

    # Missing data
    def test_case7_get_df_measure(self):

        # Calculate
        df = core.DataWrangl.get_df_measure(
            pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_indosemeasures_v1.csv')),
            config.indose_measure_params[5],)

        # Get manual solution and compare
        df_solution = read_solution('df_solution_get_df_measure_case7.csv')
        assert df_solution.equals(df)

    # Sum-score measures
    def test_case8_get_df_measure(self):
        ''' Scores are read from the column created by calc_sum_scores(); the EBI items sum to ebi_score,
            so the result must match case1
        '''

        # Calculate
        df_redcap, df_missingitems = core.DataWrangl.calc_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            measure_param = {
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'sum_score': {
                    'col_items': EBI_ITEMS,
                    'reverse_items': [],
                    'min_score': 0,
                    'max_score': 100,},})

        df = core.DataWrangl.get_df_measure(
            df_redcap = df_redcap,
            measure_param = {
                'instrument': 'EBI',
                'timing': 'post_dose',
                'measure': 'EBI',
                'col_complete': 'ebi_complete',
                'is_sum_score': True,})

        # Get manual solution & compare
        df_solution = read_solution('df_solution_get_df_measure_case1.csv')
        assert df_solution.equals(df)


class GetDfVitalsTests(DataWranglTests):

    def test_case0_get_df_vitals(self):

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_vitalsonly_case0.csv'))
        df_redcap = to_current_vitals_format(df_redcap)
        df = core.DataWrangl.get_df_vitals(df_redcap, config.vitals_measure_params[0],)

        # Get manual solution
        df_solution = read_solution('df_solution_get_df_vitals_case0.csv')
        assert df_solution.equals(df)

    def test_case1_get_df_vitals(self):

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_vitalsonly_case0.csv'))
        df_redcap = to_current_vitals_format(df_redcap)
        df = core.DataWrangl.get_df_vitals(df_redcap, config.vitals_measure_params[17],)

        # Get manual solution
        df_solution = read_solution('df_solution_get_df_vitals_case1.csv')
        assert df_solution.equals(df)

    # Missing data
    def test_case2_get_df_vitals(self):

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_vitalsonly_case1.csv'))
        df_redcap = to_current_vitals_format(df_redcap)
        df = core.DataWrangl.get_df_vitals(df_redcap, config.vitals_measure_params[17],)
        df['pID'] = df['pID'].astype('int64') # due to missing data pIDs are read as floats, converting type

        # Get manual solution
        df_solution = read_solution('df_solution_get_df_vitals_case2.csv')
        assert df_solution.equals(df)


class CalcScoresTests(DataWranglTests):
    ''' Fixtures name the sum score column 'tadaa', calc_sum_scores() names it 'tadaa_LOCAL_SCORE' '''

    def test_case0_calc_sum_scores(self):
        ''' Case of not summing anything '''

        df, df_missingitems = core.DataWrangl.calc_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            measure_param = ebi_sum_score_param(col_items=[]))
        df = df.rename(columns={'tadaa_LOCAL_SCORE': 'tadaa'})

        # Get manual solution & compare
        df_solution = read_solution('df_solution_calc_scores_case0.csv')
        assert df_solution.equals(df)

    def test_case1_calc_sum_scores(self):
        ''' Intended case '''

        # Calculate
        df, df_missingitems = core.DataWrangl.calc_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv')),
            measure_param = ebi_sum_score_param(col_items=EBI_ITEMS))
        df = df.rename(columns={'tadaa_LOCAL_SCORE': 'tadaa'})

        # Get manual solution & compare
        df_solution = read_solution('df_solution_calc_scores_case1.csv')
        assert df_solution.equals(df)

    def test_case2_calc_sum_scores(self):
        ''' Intended case, different data '''

        # Calculate
        df, df_missingitems = core.DataWrangl.calc_sum_scores(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v2.csv')),
            measure_param = ebi_sum_score_param(col_items=EBI_ITEMS))
        df = df.rename(columns={'tadaa_LOCAL_SCORE': 'tadaa'})

        # Get manual solution & compare
        df_solution = read_solution('df_solution_calc_scores_case2.csv')
        assert df_solution.equals(df)

    def test_case3_calc_sum_scores(self):
        ''' Missing data case; rows with missing items get a NaN score and are listed in df_missingitems '''

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv'))
        df_redcap.iloc[0, 2] = math.nan
        df_redcap.iloc[1, 2] = None
        df_redcap.iloc[1, 3] = None

        # Calculate
        df, df_missingitems = core.DataWrangl.calc_sum_scores(
            df_redcap = df_redcap,
            measure_param = ebi_sum_score_param(col_items=EBI_ITEMS))
        df = df.rename(columns={'tadaa_LOCAL_SCORE': 'tadaa'})

        # Get manual solution & compare
        df_solution = read_solution('df_solution_calc_scores_case3.csv')
        assert df_solution.equals(df)
        assert df_missingitems.missing_items.tolist() == [['ebi_1'], ['ebi_1', 'ebi_2']]

    # Reverse scoring
    def test_case4_calc_sum_scores(self):

        measure_param = {
            'instrument': 'mock',
            'timing': 'change',
            'col_complete': 'complete',
            'measure': 'mock',
            'sum_score': {
                'col_items': [f'mock_q{idx}' for idx in range(1,6)],
                'reverse_items': ['mock_q1', 'mock_q3', 'mock_q5',],
                'min_score': 0,
                'max_score': 100,},}

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
            'mock_LOCAL_SCORE': [500.0, 0, (70+70+75+50+10), (90*3+10*2)]
        })

        # Calculate & compare to solution
        df, df_missingitems = core.DataWrangl.calc_sum_scores(df, measure_param)
        assert df_solution.equals(df)

    def test_case5_calc_sum_scores(self):

        measure_param = {
            'instrument': 'mock',
            'timing': 'change',
            'col_complete': 'complete',
            'measure': 'mock',
            'sum_score': {
                'col_items': [f'mock_q{idx}' for idx in range(1,6)],
                'reverse_items': ['mock_q1', 'mock_q2'],
                'min_score': 1,
                'max_score': 5,},}

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
            'mock_LOCAL_SCORE': [25.0, 5, (3+3+2+5+2), (1+4+3+1+1)]
        })

        # Calculate & compare to solution
        df, df_missingitems = core.DataWrangl.calc_sum_scores(df, measure_param)
        assert df_solution.equals(df)

    def test_case6_calc_sum_scores(self):

        measure_param = {
            'instrument': 'mock',
            'timing': 'change',
            'col_complete': 'complete',
            'measure': 'mock',
            'sum_score': {
                'col_items': [f'mock_q{idx}' for idx in range(1,6)],
                'reverse_items': ['mock_q1', 'mock_q2'],
                'min_score': 0,
                'max_score': 5,},}

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
            'mock_LOCAL_SCORE': [25.0, 0, (2+5+2+5+2), (0+3+3+0+1)]
        })

        # Calculate & compare to solution
        df, df_missingitems = core.DataWrangl.calc_sum_scores(df, measure_param)
        assert df_solution.equals(df)

    def test_case7_calc_sum_scores(self):
        ''' Sum score column must not exist yet, e.g. when calc_sum_scores() is called twice '''

        df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_v1.csv'))
        df_redcap, df_missingitems = core.DataWrangl.calc_sum_scores(df_redcap, ebi_sum_score_param(col_items=EBI_ITEMS))

        with self.assertRaises(AssertionError):
            core.DataWrangl.calc_sum_scores(df_redcap, ebi_sum_score_param(col_items=EBI_ITEMS))


class ReplaceTpsTests(DataWranglTests):

    def test_case0_replace_tps(self):
        ''' Repeat rows replace the original row with the same (pID, measure, time) '''

        df = pd.DataFrame({
            'pID': [1, 1, 1, 1, 1],
            'tp': ['A0', 'A0', 'A0_R1', 'bsl', 'bsl_R1'],
            'measure': ['VITALS_hr', 'VITALS_hr', 'VITALS_hr', 'MADRS', 'MADRS'],
            'time': [0, 30, 30, math.nan, math.nan],
            'score': [60.0, 70.0, 75.0, 20.0, 22.0],})

        df, df_replaced_rows, df_duplicate_rows = core.DataWrangl.replace_tps(
            df, repeated_tps={'A0_R1': 'A0', 'bsl_R1': 'bsl'})

        df_solution = pd.DataFrame({
            'pID': [1, 1, 1],
            'tp': ['A0', 'A0', 'bsl'],
            'measure': ['VITALS_hr', 'VITALS_hr', 'MADRS'],
            'time': [0, 30, math.nan],
            'score': [60.0, 75.0, 22.0],})

        assert df_solution.equals(df)
        assert df_replaced_rows.score.tolist() == [20.0, 70.0]
        assert df_duplicate_rows.empty

    def test_case1_replace_tps(self):
        ''' Repeat rows without an original row are relabeled; other participants and the input df are unchanged '''

        df_input = pd.DataFrame({
            'pID': [1, 2],
            'tp': ['A0', 'A0_R1'],
            'measure': ['MADRS', 'MADRS'],
            'time': [math.nan, math.nan],
            'score': [20.0, 25.0],})
        df_input_copy = df_input.copy()

        df, df_replaced_rows, df_duplicate_rows = core.DataWrangl.replace_tps(
            df_input, repeated_tps={'A0_R1': 'A0'})

        df_solution = pd.DataFrame({
            'pID': [1, 2],
            'tp': ['A0', 'A0'],
            'measure': ['MADRS', 'MADRS'],
            'time': [math.nan, math.nan],
            'score': [20.0, 25.0],})

        assert df_solution.equals(df)
        assert df_replaced_rows.empty
        assert list(df_replaced_rows.columns) == list(df.columns)
        assert df_duplicate_rows.empty
        assert df_input.equals(df_input_copy)

    def test_case2_replace_tps(self):
        ''' Differences between the original and repeat row in non-value columns raise a warning '''

        df = pd.DataFrame({
            'pID': [1, 1],
            'tp': ['A0', 'A0_R1'],
            'measure': ['MADRS', 'MADRS'],
            'time': [math.nan, math.nan],
            'rater': ['fac', 'pat'],
            'score': [20.0, 22.0],})

        with self.assertWarns(UserWarning) as warning:
            df, df_replaced_rows, df_duplicate_rows = core.DataWrangl.replace_tps(
                df, repeated_tps={'A0_R1': 'A0'})

        assert "column 'rater'" in str(warning.warning)
        assert df.rater.tolist() == ['pat']
        assert df.score.tolist() == [22.0]
        assert df_replaced_rows.rater.tolist() == ['fac']

    def test_case3_replace_tps(self):
        ''' No warning if non-value columns match or are both NaN; score and delta_score may differ '''

        df = pd.DataFrame({
            'pID': [1, 1],
            'tp': ['A0', 'A0_R1'],
            'measure': ['MADRS', 'MADRS'],
            'time': [math.nan, math.nan],
            'rater': ['fac', 'fac'],
            'comment': [math.nan, math.nan],
            'score': [20.0, 22.0],
            'delta_score': [-5.0, -3.0],})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            df, df_replaced_rows, df_duplicate_rows = core.DataWrangl.replace_tps(
                df, repeated_tps={'A0_R1': 'A0'})

        assert not [w for w in caught if 'Discrepancy' in str(w.message)]
        assert df.score.tolist() == [22.0]
        assert df.delta_score.tolist() == [-3.0]

    def test_case4_replace_tps(self):
        ''' Rows sharing (pID, tp, measure, time) after replacement are returned as duplicates '''

        df_input = pd.DataFrame({
            'pID': [1, 1, 2],
            'tp': ['bsl', 'bsl', 'bsl'],
            'measure': ['MADRS', 'MADRS', 'MADRS'],
            'time': [math.nan, math.nan, math.nan],
            'score': [20.0, 21.0, 30.0],})

        df, df_replaced_rows, df_duplicate_rows = core.DataWrangl.replace_tps(df_input, repeated_tps={})

        assert df_input.equals(df)
        assert df_replaced_rows.empty
        assert df_duplicate_rows.pID.tolist() == [1, 1]
        assert df_duplicate_rows.score.tolist() == [20.0, 21.0]


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

        ### Change both delta_from inputs

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
    ''' Fixtures predate the 'n' column of get_df_observed(), so it is dropped before comparing '''

    def test_case0_get_df_observed(self):

        # Calculate
        df = core.Analysis.get_df_observed(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_v0.csv')),)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_get_df_observed_case0.csv'))

        # Compare
        assert df_solution.equals(df.drop(columns='n'))

    def test_case1_get_df_observed(self):

        # Calculate
        df = core.Analysis.get_df_observed(
            df_master = pd.read_csv(os.path.join(folders.fixtures_in, 'df_master_v1.csv')),)

        # Get manual solution
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_get_df_observed_case1.csv'))

        # Compare
        assert df_solution.equals(df.drop(columns='n'))


class GetDfTpNdaysTests(AnalysisTests):
    ''' Fixtures were generated with the mean number of days '''

    def test_case0_get_df_tp_ndays(self):

        # Calculate
        df = core.DataWrangl.get_df_tp_ndays(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_wdates_case0.csv')),
            col_date = 'date',
            stat = 'mean')
        df.reset_index(drop=True, inplace=True)

        # Get manual solution & compare
        df_solution = pd.read_csv(os.path.join(folders.fixtures_out, 'df_solution_ get_df_tp_ndays_case0.csv'))
        assert df_solution.equals(df)

    def test_case1_get_df_tp_ndays(self):

        # Calculate
        df = core.DataWrangl.get_df_tp_ndays(
            df_redcap = pd.read_csv(os.path.join(folders.fixtures_in, 'df_redcap_wdates_case1.csv')),
            col_date = 'date',
            stat = 'mean')
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

        ### Check Pearson correlation for A21-bsl_vs_A0 case
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

        ### Check Spearman correlation for A21-bsl_vs_A0 case
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

        ### Check Kendall correlation for A21-bsl_vs_A0 case
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

    @staticmethod
    def get_df_corr():
        ''' Small wide-format df with hand-checkable correlations of x with the other columns:
            y: pearson/spearman r=0.8 (p=0.104), kendall tau=0.6 (p=0.233)
            z: pearson/spearman r=-0.8 (p=0.104), kendall tau=-0.6 (p=0.233)
        '''
        return pd.DataFrame({
            'pID': [1, 2, 3, 4, 5],
            'x': [1, 2, 3, 4, 5],
            'y': [1, 3, 2, 5, 4],
            'z': [5, 3, 4, 1, 2],})

    def test_case2_get_corrmat(self):
        ''' Coefficients and p-values of each method; rows are yvars, columns are xvars '''

        expected = {
            'pearson': ([0.8, -0.8], [0.104, 0.104]),
            'spearman': ([0.8, -0.8], [0.104, 0.104]),
            'kendall': ([0.6, -0.6], [0.233, 0.233]),}

        for method, (coeffs, pvalues) in expected.items():
            df_coeffs, df_pvalues = core.Analysis.get_corrmats(
                df = self.get_df_corr(),
                xvars = ['x'],
                yvars = ['y', 'z'],
                methods = [method],
                draw = False)

            assert pd.DataFrame({'x': coeffs}, index=['y', 'z']).equals(df_coeffs), method
            assert pd.DataFrame({'x': pvalues}, index=['y', 'z']).equals(df_pvalues), method

    def test_case3_get_corrmat(self):
        ''' A variable present in both xvars and yvars has coefficient 1 and p-value 0 with itself '''

        df_coeffs, df_pvalues = core.Analysis.get_corrmats(
            df = self.get_df_corr(),
            xvars = ['x', 'y'],
            yvars = ['x', 'y'],
            methods = ['pearson'],
            draw = False)

        assert pd.DataFrame({'x': [1.0, 0.8], 'y': [0.8, 1.0]}, index=['x', 'y']).equals(df_coeffs)
        assert pd.DataFrame({'x': [0.0, 0.104], 'y': [0.104, 0.0]}, index=['x', 'y']).equals(df_pvalues)

    def test_case4_get_corrmat(self):
        ''' Missing values are dropped pairwise: the extra row is used for x-z but not for x-y '''

        df = pd.concat([
            self.get_df_corr(),
            pd.DataFrame({'pID': [6], 'x': [6], 'y': [math.nan], 'z': [0]})], ignore_index=True)

        df_coeffs, df_pvalues = core.Analysis.get_corrmats(
            df = df,
            xvars = ['x'],
            yvars = ['y', 'z'],
            methods = ['pearson'],
            draw = False)

        # x-z with all 6 rows: r = -31/35
        assert pd.DataFrame({'x': [0.8, -0.886]}, index=['y', 'z']).equals(df_coeffs)
        assert pd.DataFrame({'x': [0.104, 0.019]}, index=['y', 'z']).equals(df_pvalues)

    def test_case5_get_corrmat(self):
        ''' With several methods, each is saved to CSV, nothing is drawn if draw=False, and the last method is returned '''

        with tempfile.TemporaryDirectory() as dir_out, mock.patch.object(core.Plots, 'draw_corrmat') as draw_corrmat:
            df_coeffs, df_pvalues = core.Analysis.get_corrmats(
                df = self.get_df_corr(),
                xvars = ['x'],
                yvars = ['y', 'z'],
                methods = ['pearson', 'kendall'],
                draw = False,
                dir_out = dir_out,
                fname_out = 'corr')

            assert sorted(os.listdir(dir_out)) == [
                'corr_kendall_coeffs.csv', 'corr_kendall_pvalues.csv',
                'corr_pearson_coeffs.csv', 'corr_pearson_pvalues.csv']
            df_pearson_coeffs = pd.read_csv(os.path.join(dir_out, 'corr_pearson_coeffs.csv'), index_col=0)
            df_kendall_pvalues = pd.read_csv(os.path.join(dir_out, 'corr_kendall_pvalues.csv'), index_col=0)

        draw_corrmat.assert_not_called()
        assert pd.DataFrame({'x': [0.8, -0.8]}, index=['y', 'z']).equals(df_pearson_coeffs)
        assert pd.DataFrame({'x': [0.233, 0.233]}, index=['y', 'z']).equals(df_kendall_pvalues)
        assert pd.DataFrame({'x': [0.6, -0.6]}, index=['y', 'z']).equals(df_coeffs)
        assert pd.DataFrame({'x': [0.233, 0.233]}, index=['y', 'z']).equals(df_pvalues)

    def test_case6_get_corrmat(self):
        ''' Unless draw=False, each method is drawn with its own matrices, title info and formatting kwargs '''

        with mock.patch.object(core.Plots, 'draw_corrmat') as draw_corrmat:
            core.Analysis.get_corrmats(
                df = self.get_df_corr(),
                xvars = ['x'],
                yvars = ['y', 'z'],
                methods = ['pearson', 'kendall'],
                xlabel = 'baseline')

        assert draw_corrmat.call_count == 2
        for call, method, coeffs in zip(draw_corrmat.call_args_list, ['pearson', 'kendall'], [[0.8, -0.8], [0.6, -0.6]]):
            assert call.kwargs['method'] == method
            assert call.kwargs['corr_info'] == f'{method.upper()} correlation (n=5)'
            assert call.kwargs['xlabel'] == 'baseline'
            assert pd.DataFrame({'x': coeffs}, index=['y', 'z']).equals(call.kwargs['df_coeffs'])

    def test_case7_get_corrmat(self):
        ''' Variables must be columns of df '''

        with self.assertRaises(AssertionError):
            core.Analysis.get_corrmats(
                df = self.get_df_corr(),
                xvars = ['x'],
                yvars = ['not_a_column'],
                methods = ['pearson'],
                draw = False)


class GetMissingScoresTests(AnalysisTests):

    @staticmethod
    def get_df_master():
        ''' Long-form df where each pID is complete for some measures and timepoints only:
            - MADRS (no time): pID 1 complete; pID 2 has a NaN score at bsl and no A0 row;
              pID 3 has two bsl rows and no A0 row
            - VITALS_hr (time): pID 1 has A0 at time 30 and 60; pID 2 only at time 60; pID 3 has no rows
        '''
        return pd.DataFrame({
            'pID': [1, 1, 2, 3, 3, 1, 1, 2],
            'tp': ['bsl', 'A0', 'bsl', 'bsl', 'bsl', 'A0', 'A0', 'A0'],
            'measure': ['MADRS']*5 + ['VITALS_hr']*3,
            'time': [math.nan]*5 + [30, 60, 60],
            'score': [20.0, 15.0, math.nan, 18.0, 19.0, 70.0, 72.0, 71.0],})

    def test_case0_get_missing_scores(self):
        ''' Measure without time: NaN, absent and duplicate scores are all missing '''

        df = core.Analysis.get_missing_scores(
            df_master = self.get_df_master(),
            measure_params = [{'measure': 'MADRS', 'tps': ['bsl', 'A0'], 'has_time': False}])

        assert list(df.columns) == ['pID', 'tp', 'measure', 'time']
        assert df[['pID', 'tp', 'measure']].values.tolist() == [
            [2, 'bsl', 'MADRS'],
            [2, 'A0', 'MADRS'],
            [3, 'bsl', 'MADRS'],
            [3, 'A0', 'MADRS']]
        assert df.time.isna().all()

    def test_case1_get_missing_scores(self):
        ''' Measure with time: only scores at measure_param['time'] count; every pID of df_master is expected '''

        df = core.Analysis.get_missing_scores(
            df_master = self.get_df_master(),
            measure_params = [{'measure': 'VITALS_hr', 'tps': ['A0'], 'has_time': True, 'time': 30}])

        assert df[['pID', 'tp', 'measure']].values.tolist() == [
            [2, 'A0', 'VITALS_hr'],
            [3, 'A0', 'VITALS_hr']]
        assert df.time.tolist() == [30, 30]

    def test_case2_get_missing_scores(self):
        ''' Several measures are checked in order; complete participants add no rows '''

        df = core.Analysis.get_missing_scores(
            df_master = self.get_df_master(),
            measure_params = [
                {'measure': 'VITALS_hr', 'tps': ['A0'], 'has_time': True, 'time': 60},
                {'measure': 'MADRS', 'tps': ['bsl'], 'has_time': False},])

        assert df[['pID', 'tp', 'measure']].values.tolist() == [
            [3, 'A0', 'VITALS_hr'],
            [2, 'bsl', 'MADRS'],
            [3, 'bsl', 'MADRS']]

        df = core.Analysis.get_missing_scores(
            df_master = self.get_df_master().loc[lambda df: df.pID == 1],
            measure_params = [{'measure': 'MADRS', 'tps': ['bsl', 'A0'], 'has_time': False}])

        assert df.empty
        assert list(df.columns) == ['pID', 'tp', 'measure', 'time']

    def test_case3_get_missing_scores(self):
        ''' has_time must be True or False '''

        with self.assertRaises(AssertionError):
            core.Analysis.get_missing_scores(
                df_master = self.get_df_master(),
                measure_params = [{'measure': 'MADRS', 'tps': ['bsl'], 'has_time': None}])

    def test_case4_get_missing_scores(self):
        ''' Output is saved to CSV if dir_out and fname_out are given '''

        with tempfile.TemporaryDirectory() as dir_out:
            df = core.Analysis.get_missing_scores(
                df_master = self.get_df_master(),
                measure_params = [{'measure': 'MADRS', 'tps': ['bsl', 'A0'], 'has_time': False}],
                dir_out = dir_out,
                fname_out = 'missing_scores.csv')
            df_saved = pd.read_csv(os.path.join(dir_out, 'missing_scores.csv'))

        assert df_saved[['pID', 'tp', 'measure']].values.tolist() == df[['pID', 'tp', 'measure']].values.tolist()
