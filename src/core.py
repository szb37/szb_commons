'''
Functions shared across clinical trial analysis codebases.

Data typically flows from a REDCap export (df_redcap, one row per participant and timepoint) to a
long-form master df (df_master, one row per participant, timepoint, measure and time), which is then
checked, analyzed and plotted. Functions are static methods grouped into classes:
    - DataWrangl: clean REDCap exports and convert them to long-form data
    - Analysis: descriptive statistics, correlations and missing data
    - Plots: figures
    - CheckDf: sanity checks of df_master
    - Helpers: small utilities used by the classes above

Columns of df_master: pID, tp (timepoint), timing, instrument, measure, time (NaN for measures
without time), score, and optionally delta_score and condition.
See config.py for the expected format of measure_param dictionaries.
'''

import commons_codebase.src.config as commons_config
from statistics import mean, stdev
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import warnings
import math
import os

# Silence pandas/seaborn deprecation notices; note this applies to all code that imports this module
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DataWrangl:
    ''' Functions for common data wrangling tasks '''

    @staticmethod
    def clean_df_redcap(df_redcap: pd.DataFrame, rename_cols: dict = {}, rename_tps: dict = {}, rm_tps: list = []) -> pd.DataFrame:
        ''' Rename columns, rename timepoints and remove spurious timepoints of a REDCap export.
            If rename_tps is given, timepoints that remain outside its values are printed.

            Args:
                - df_redcap (pd.DataFrame): raw REDCap export dataframe
                - rename_cols (dict, optional): mapping of old to new column names,
                    e.g. commons_config.redcap_cols_to_master_cols
                - rename_tps (dict, optional): mapping of old to new timepoint names,
                    e.g. {'a00_dosing_session_arm_1': 'A0'}
                - rm_tps (list, optional): timepoints to remove; matched after renaming

            Returns:
                - df_redcap (pd.DataFrame): cleaned REDCap dataframe
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(rename_cols, dict)
        assert isinstance(rename_tps, dict)
        assert isinstance(rm_tps, list)

        df_redcap = df_redcap.rename(columns=rename_cols)
        df_redcap['tp'] = df_redcap['tp'].replace(rename_tps)
        df_redcap = df_redcap.loc[~df_redcap.tp.isin(rm_tps)]

        ### Report tps that are not among the renamed (i.e. expected) tps, e.g. typos in rename_tps
        if rename_tps:
            valid_tps = set(rename_tps.values())
            invalid_tps = list(set(df_redcap.loc[~df_redcap.tp.isin(valid_tps), 'tp']))
            if invalid_tps:
                print(f'Non-defined tps found in df_redcap: {invalid_tps}')

        return df_redcap

    @staticmethod
    def clean_pIDs(df_redcap: pd.DataFrame, n_prefix_chars: int = 0) -> pd.DataFrame:
        ''' Convert pIDs to int. Rows whose pID can not be converted are printed and removed.

            Args:
                - df_redcap (pd.DataFrame): REDCap dataframe with a 'pID' column
                - n_prefix_chars (int, optional): number of leading characters removed from each pID
                    before conversion, e.g. with n_prefix_chars=3 'PDP2001' becomes 2001

            Returns:
                - df_redcap (pd.DataFrame): dataframe with int pIDs
        '''

        df_redcap = df_redcap.copy()
        if n_prefix_chars > 0:
            df_redcap['pID'] = df_redcap['pID'].astype(str).str[n_prefix_chars:]

        ### Print & delete non-convertible pIDs; errors='coerce' turns them into NaN
        pID_numeric = pd.to_numeric(df_redcap['pID'], errors='coerce')
        invalid_pids = df_redcap.loc[pID_numeric.isna(), 'pID'].unique()
        if len(invalid_pids) > 0:
            print('Unique pIDs that could not be converted to int (deleting rows):')
            for invalid_pid in invalid_pids:
                print(f'\t{invalid_pid}')

        df_redcap = df_redcap.loc[pID_numeric.notna()].copy()
        df_redcap['pID'] = pID_numeric[pID_numeric.notna()].astype(int)

        return df_redcap

    @staticmethod
    def replace_tps(df: pd.DataFrame, repeated_tps: dict = {}) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ''' Replace original timepoints with their repeats in a long-form dataframe.

            For each {new_tp: old_tp} pair in repeated_tps, a row at new_tp overwrites the row at
            old_tp with the same (pID, measure, time); missing times are matched to missing times.
            If there is no such original row, the repeat row is relabeled to old_tp.
            Discrepancies between the two rows in any column other than
            pID/measure/time/tp/score/delta_score raise a warning.

            Args:
                - df (pd.DataFrame): long-form dataframe with columns 'pID', 'tp', 'measure', 'time',
                    'score' (and optionally 'delta_score')
                - repeated_tps (dict): mapping of {new_tp: old_tp},
                    e.g. {'bsl_R1': 'bsl', 'A0_R1': 'A0'}

            Returns:
                - df (pd.DataFrame): copy of the input dataframe after replacement, without any new_tp rows
                - df_replaced_rows (pd.DataFrame): original rows (pre-replacement state) that were
                    overwritten by a repeat row; empty if none
                - df_duplicate_rows (pd.DataFrame): rows sharing a (pID, tp, measure, time)
                    combination after replacement; empty if none
        '''

        df = df.copy()
        ignore_cols = {'pID', 'measure', 'time', 'tp', 'score', 'delta_score'}
        compare_cols = [col for col in df.columns if col not in ignore_cols]
        replaced_rows = []

        for new_tp, old_tp in repeated_tps.items():
            new_rows = df.loc[df.tp == new_tp]
            for new_idx, new_row in new_rows.iterrows():
                ### Find the original row of the same participant, measure and time
                pID = new_row['pID']
                measure = new_row['measure']
                time = new_row['time']
                # NaN never equals NaN, so missing times have to be matched with isna()
                is_same_time = df.time.isna() if pd.isna(time) else (df.time == time)
                old_mask = (
                    (df.pID == pID) &
                    (df.measure == measure) &
                    (df.tp == old_tp) &
                    is_same_time)
                old_matches = df.loc[old_mask]

                if old_matches.empty:
                    ### No original row to replace; just relabel the repeat as the original tp
                    df.at[new_idx, 'tp'] = old_tp
                    continue

                ### Check for discrepancies in non-value columns
                old_row = old_matches.iloc[0]
                for col in compare_cols:
                    old_val = old_row[col]
                    new_val = new_row[col]
                    if pd.isna(old_val) and pd.isna(new_val):
                        continue
                    if old_val != new_val:
                        warnings.warn(
                            f"Discrepancy for pID={pID}, measure={measure}, time={time}, column '{col}': "
                            f"{new_tp}={new_val!r} vs {old_tp}={old_val!r}")

                ### Keep the pre-replacement row, then overwrite it with the repeat (keeping tp=old_tp)
                replaced_rows.append(old_row.copy())
                replacement = new_row.copy()
                replacement['tp'] = old_tp
                # Assign all columns at once; .values aligns by position, new_row has the same column order as df
                df.loc[old_matches.index[0], df.columns] = replacement.values

            ### Drop the repeat rows now that their data has been merged into old_tp
            df = df.loc[df.tp != new_tp].reset_index(drop=True)

        ### Collect replaced rows and any duplicates that remain after replacement
        if replaced_rows:
            df_replaced_rows = pd.DataFrame(replaced_rows).sort_values(['pID', 'measure', 'tp']).reset_index(drop=True)
        else:
            df_replaced_rows = pd.DataFrame(columns=df.columns)

        is_duplicate = df.duplicated(subset=['pID', 'tp', 'measure', 'time'], keep=False)
        df_duplicate_rows = df.loc[is_duplicate].sort_values(['pID', 'measure', 'tp']).reset_index(drop=True)

        return df, df_replaced_rows, df_duplicate_rows

    @staticmethod
    def get_df_tp_ndays(df_redcap: pd.DataFrame, col_date: str, stat: str = 'median', **save) -> pd.DataFrame:
        ''' Get df of the typical number of days between baseline and each timepoint.
            Only rows with study_visit_completion_record_complete==2 are used. A participant
            contributes to a timepoint only if they have exactly one dated row at both baseline and
            that timepoint.

            Args:
                - df_redcap (pd.DataFrame): REDCap export df with 'pID' and 'tp' columns
                - col_date (str): column of visit dates in df_redcap
                - stat (str): how days are summarized across participants; 'median' or 'mean'
                - save (dict; optional): keys dir_out and fname_out determine where output is saved

            Returns:
                - df_tp_ndays (pd.DataFrame): columns tp, ndays (rounded days since baseline) and
                    n (number of participants), sorted by ndays
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(col_date, str)
        assert stat in ('median', 'mean')

        summarize = np.median if stat == 'median' else np.mean

        df = df_redcap.rename(columns={col_date: 'date'})
        df = df.loc[df.study_visit_completion_record_complete == 2]
        df = df[['pID', 'tp', 'date']].dropna()
        df.date = pd.to_datetime(df.date)
        # After reset_index, index labels equal row positions, so they can be used with iloc below
        df.reset_index(drop=True, inplace=True)
        cidx_date = df.columns.get_loc('date')

        # Baseline is the reference, so its offset is 0 by definition
        df_tp_ndays_rows = [['bsl', 0, math.nan]]

        ### For each tp, collect the days between baseline and tp of every participant, then summarize
        for tp in df.tp.unique():
            if tp == 'bsl':
                continue

            ndays = []
            for pID in df.pID.unique():
                row_bsl = df.loc[(df.pID == pID) & (df.tp == 'bsl')]
                row_tp = df.loc[(df.pID == pID) & (df.tp == tp)]

                ### Ignore participant if there is not exactly 1 row for either tp or bsl
                if row_bsl.shape[0] == 1 and row_tp.shape[0] == 1:
                    ndays.append(
                        (df.iloc[row_tp.index[0], cidx_date] - df.iloc[row_bsl.index[0], cidx_date]).days)

            df_tp_ndays_rows.append([tp, round(summarize(ndays)), len(ndays)])

        df_tp_ndays = pd.DataFrame(columns=['tp', 'ndays', 'n'], data=df_tp_ndays_rows)
        df_tp_ndays.sort_values(by='ndays', inplace=True)

        if save:
            df_tp_ndays.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_tp_ndays

    @staticmethod
    def get_df_measure(df_redcap: pd.DataFrame, measure_param: dict, cols_to_keep: list[str] = commons_config.cols_to_keep, **save) -> pd.DataFrame:
        ''' Returns all completed scores of a given measure in a long-format df.
            Rows with a missing pID, tp or score are removed.

            Args:
                - df_redcap (pd.DataFrame): REDCap export; for sum-score measures, calc_sum_scores()
                    must be called on it first
                - measure_param (dict): defines the measure, see config.py for the expected format. Uses keys:
                    - 'instrument', 'measure', 'timing': values of the respective columns in the returned df
                    - 'col_complete': completion column in df_redcap; only rows where it equals 2 are kept;
                        if None, completion is not checked
                    - 'is_sum_score' (optional): if True, scores are read from the column created by
                        calc_sum_scores(); otherwise from 'col_score'
                    - 'col_score': score column in df_redcap; only needed if is_sum_score is not True
                    - 'time' (optional): value of the 'time' column in the returned df; NaN if not given
                - cols_to_keep (list[str]): columns of df_redcap to keep in the returned df
                - save (dict; optional): keys dir_out and fname_out determine where output is saved

            Returns:
                - df (pd.DataFrame): long-format df with columns cols_to_keep + timing, instrument,
                    measure, time, score
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(measure_param, dict)
        assert all(field in measure_param for field in ['instrument', 'measure', 'timing', 'col_complete'])
        assert isinstance(cols_to_keep, list)

        if measure_param.get('is_sum_score', False):
            col_score = Helpers.get_col_sum_score(measure_param)
        else:
            col_score = measure_param['col_score']
        col_complete = measure_param['col_complete']

        df = df_redcap.copy()
        if col_complete is not None:
            df = df.loc[df[col_complete] == 2]
        df = df[cols_to_keep + [col_score]].rename(columns={col_score: 'score'})

        df['instrument'] = measure_param['instrument']
        df['measure'] = measure_param['measure']
        df['timing'] = measure_param['timing']
        df['time'] = measure_param.get('time', math.nan)

        df = df.dropna(subset=['pID', 'tp', 'score'])
        df['score'] = df['score'].astype('float64')
        df['time'] = df['time'].astype('float64')
        df = df[cols_to_keep + ['timing', 'instrument', 'measure', 'time', 'score']]
        df.reset_index(inplace=True, drop=True)

        if save:
            df.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df

    @staticmethod
    def calc_sum_scores(df_redcap: pd.DataFrame, measure_param: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        ''' Calculates the sum score of a measure and stores it in a new column of df_redcap.

            The new column is named f"{measure_param['measure']}_LOCAL_SCORE" and must not exist yet.
            Only completed rows (col_complete==2) are scored. Reverse items are scored as
            (min_score + max_score - item) in the sum; the item columns themselves are not changed.
            Rows with any missing item get a NaN sum score and are listed in df_missingitems.

            Args:
                - df_redcap (pd.DataFrame): REDCap export; the score column is added in place
                - measure_param (dict): defines the measure, see config.py for the expected format. Uses keys:
                    - 'measure': name of the measure
                    - 'col_complete': completion column in df_redcap
                    - 'sum_score': dict with keys
                        - 'col_items' (list[str]): item columns to sum
                        - 'reverse_items' (list[str]): subset of col_items that are reverse scored
                        - 'min_score', 'max_score' (int): min / max possible score of a single item

            Returns:
                - df_redcap (pd.DataFrame): input df with the sum score column added
                - df_missingitems (pd.DataFrame): columns pID, tp, measure, missing_items; one row
                    for each completed row with missing items
        '''

        col_complete = measure_param['col_complete']
        col_score = Helpers.get_col_sum_score(measure_param)
        sum_score = measure_param['sum_score']
        col_items = sum_score['col_items']
        reverse_items = sum_score['reverse_items']

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(col_items, list)
        assert all(col_item in df_redcap.columns for col_item in col_items)
        assert col_complete in df_redcap.columns
        assert col_score not in df_redcap.columns, f"Column '{col_score}' already exists in df_redcap"

        is_complete = df_redcap[col_complete] == 2

        ### Calculate sum scores, reversing items where needed
        # Items are reversed on a copy, so df_redcap keeps the original item scores.
        # (min + max - item) mirrors an item within its range, e.g. on a 1-5 scale 1 -> 5 and 4 -> 2
        items = df_redcap.loc[is_complete, col_items].copy()
        if reverse_items:
            items[reverse_items] = (sum_score['min_score'] + sum_score['max_score']) - items[reverse_items]
        df_redcap.loc[is_complete, col_score] = items.sum(axis=1)

        ### Remove sum scores with missing items; sum() above skips NaN, which would underestimate scores
        rows_df_missingitems = []
        for row in df_redcap.loc[is_complete].itertuples():
            cols_missing = [col for col in col_items if pd.isna(getattr(row, col))]
            if cols_missing:
                rows_df_missingitems.append([row.pID, row.tp, measure_param['measure'], cols_missing])
                df_redcap.loc[row.Index, col_score] = math.nan

        df_missingitems = pd.DataFrame(columns=['pID', 'tp', 'measure', 'missing_items'], data=rows_df_missingitems)
        return df_redcap, df_missingitems

    @staticmethod
    def add_delta_scores(df_master: pd.DataFrame, delta_from_tp: str = 'bsl', delta_from_time: int = 0) -> pd.DataFrame:
        ''' Add delta_score column to a long-form trial dataframe.

            For each measure the baseline is chosen based on Helpers.has_time():
                - measures without time: baseline is the row at tp == delta_from_tp for the same
                  (pID, measure); delta_score = score - baseline score for all rows of the measure
                - measures with time: baseline is the row at time == delta_from_time for the same
                  (pID, measure, tp); delta_score = score - baseline score for all rows at that tp
            Measures for which has_time can not be decided are printed and get delta_score=NaN.

            Args:
                - df_master (pd.DataFrame): long-form df of trial data
                - delta_from_tp (str): value of 'tp' that is baseline for measures without time
                - delta_from_time (int): value of 'time' that is baseline for measures with time

            Returns:
                - df_master (pd.DataFrame): copy of input df with 'delta_score' column added
        '''

        assert isinstance(df_master, pd.DataFrame)
        assert isinstance(delta_from_tp, str)
        assert isinstance(delta_from_time, int)

        df_master = df_master.copy()
        df_master['delta_score'] = math.nan

        undecided_has_time = []
        pieces = []

        ### Process each measure separately, then put the pieces back together in the original row order
        for measure, sub in df_master.groupby('measure', sort=False):
            try:
                has_time = Helpers.has_time(df_master, measure)
            except UndecidedHasTime as e:
                undecided_has_time.append(e.measure)
                pieces.append(sub)
                continue

            # Attach the matching baseline score to every row with a left merge
            if has_time:
                bsl = (sub.loc[sub['time'] == delta_from_time, ['pID', 'tp', 'score']]
                          .rename(columns={'score': '_bsl_score'}))
                merged = sub.merge(bsl, on=['pID', 'tp'], how='left')
            else:
                bsl = (sub.loc[sub['tp'] == delta_from_tp, ['pID', 'score']]
                          .rename(columns={'score': '_bsl_score'}))
                merged = sub.merge(bsl, on=['pID'], how='left')

            # merge() resets the index; restore it so sort_index() below recovers the original row order
            merged.index = sub.index
            merged['delta_score'] = merged['score'] - merged['_bsl_score']
            pieces.append(merged.drop(columns=['_bsl_score']))

        df_master = pd.concat(pieces).sort_index()

        if undecided_has_time:
            print(f"Can not decide whether measure has time: {sorted(set(undecided_has_time))}")

        return df_master

    @staticmethod
    def get_df_vitals(df_redcap: pd.DataFrame, measure_param: dict, **save) -> pd.DataFrame:
        ''' Special case of get_df_measure() to deal with the idiosyncrasies of vitals measures.
            Vitals have 1 or 2 readings, stored in columns f'{col_score}_1' and f'{col_score}_2'.
            If there are 2 readings, their mean is used as score.

            Args:
                - df_redcap (pd.DataFrame): REDCap export df
                - measure_param (dict): defines the measure, see config.py for the expected format. Uses keys:
                    - 'instrument', 'measure', 'timing', 'time': values of the respective columns in the returned df
                    - 'col_complete': completion column in df_redcap; only rows where it equals 2 are kept;
                        if None, completion is not checked
                    - 'col_score': prefix of the two reading columns in df_redcap
                - save (dict; optional): keys dir_out and fname_out determine where output is saved

            Returns:
                - df_vitals (pd.DataFrame): long-form dataframe of vitals data
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(measure_param, dict)
        assert all(field in measure_param for field in
            ['instrument', 'measure', 'timing', 'col_complete', 'col_score', 'time'])

        col_complete = measure_param['col_complete']
        df = df_redcap if col_complete is None else df_redcap.loc[df_redcap[col_complete] == 2]
        rows_vitals = []

        for row in df.itertuples():
            # Naming of the reading columns is inconsistent across trials; double check it matches the REDCap export
            score1 = getattr(row, f"{measure_param['col_score']}_1")
            score2 = getattr(row, f"{measure_param['col_score']}_2")

            if math.isnan(score1) and math.isnan(score2):
                continue  # no reading taken
            assert not math.isnan(score1), 'Second vitals reading should only exist if there was a first one'

            if math.isnan(score2):
                score = score1
            else:
                score = round((score1 + score2) / 2, 3)

            rows_vitals.append([
                row.pID,
                row.tp,
                measure_param['timing'],
                measure_param['instrument'],
                measure_param['measure'],
                measure_param['time'],
                score])

        df_vitals = pd.DataFrame(
            columns=['pID', 'tp', 'timing', 'instrument', 'measure', 'time', 'score'],
            data=rows_vitals)

        df_vitals.dropna(subset=['pID', 'tp', 'score'], inplace=True)
        df_vitals['score'] = df_vitals['score'].astype('float64')
        df_vitals['time'] = df_vitals['time'].astype('float64')
        df_vitals.reset_index(inplace=True, drop=True)

        if save:
            df_vitals.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_vitals

    @staticmethod
    def widen_master(df_master: pd.DataFrame, xvars: list[str], x_tp: str, x_use_delta: bool, yvars: list[str], y_tp: str, y_use_delta: bool) -> pd.DataFrame:
        ''' Convert long-form master df to wide-format df for correlation analysis.

            Args:
                - df_master (pd.DataFrame): long-form master df of the trial
                - xvars (list[str]): measures that become one set of columns in the wide-format df
                - x_tp (str): timepoint from which scores of xvars are taken
                - x_use_delta (bool): if True, use delta_score instead of score for xvars
                - yvars (list[str]): measures that become the other set of columns in the wide-format df
                - y_tp (str): timepoint from which scores of yvars are taken
                - y_use_delta (bool): if True, use delta_score instead of score for yvars

            Returns:
                - df (pd.DataFrame): wide-format df with a pID column and one column per measure;
                    if a pID has several rows for a measure, their mean is used
        '''

        assert isinstance(df_master, pd.DataFrame)
        for axis_vars, axis_tp in [(xvars, x_tp), (yvars, y_tp)]:
            assert isinstance(axis_vars, list)
            assert all(isinstance(measure, str) for measure in axis_vars)
            assert isinstance(axis_tp, str)

        df = df_master.loc[
            ((df_master.tp == x_tp) & (df_master.measure.isin(xvars))) |
            ((df_master.tp == y_tp) & (df_master.measure.isin(yvars)))].copy()

        ### Copy delta scores into 'score' where requested, so a single column can be pivoted
        is_x = (df.tp == x_tp) & df.measure.isin(xvars)
        is_y = (df.tp == y_tp) & df.measure.isin(yvars)

        if x_use_delta:
            df.loc[is_x, 'score'] = df.loc[is_x, 'delta_score']
        if y_use_delta:
            df.loc[is_y, 'score'] = df.loc[is_y, 'delta_score']

        # One row per pID, one column per measure; duplicate (pID, measure) rows are averaged
        df = pd.pivot_table(df, index=['pID'], columns='measure', values='score', dropna=False)
        df.reset_index(inplace=True)

        return df


class Analysis:
    ''' Functions for data analysis '''

    @staticmethod
    def get_df_observed(df_master: pd.DataFrame, digits: int = 3, **save) -> pd.DataFrame:
        ''' Creates a dataframe with the observed mean and SD of all measures at every tp.
            Missing scores are ignored; measure/tp combinations with fewer than 3 scores are skipped.

            Args:
                - df_master (pd.DataFrame): long-form master dataframe containing all data
                - digits (int): number of digits mean and SD are rounded to
                - save (dict; optional): keys dir_out and fname_out determine where output is saved

            Returns:
                - df_observed (pd.DataFrame): columns measure, tp, n, mean, sd
        '''

        measures = df_master.measure.unique().tolist()
        tps = df_master.tp.unique().tolist()
        rows_observed = []

        for measure, tp in itertools.product(measures, tps):
            scores = df_master.loc[(df_master.measure == measure) & (df_master.tp == tp)].score.dropna().tolist()
            if len(scores) < 3:
                continue

            rows_observed.append([
                measure,
                tp,
                len(scores),
                round(mean(scores), digits),
                round(stdev(scores), digits)])

        df_observed = pd.DataFrame(columns=['measure', 'tp', 'n', 'mean', 'sd'], data=rows_observed)

        if save:
            df_observed.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_observed

    @staticmethod
    def get_corrmats(df: pd.DataFrame, xvars: list[str], yvars: list[str], methods: list[str] = commons_config.corr_methods, **kwargs):
        ''' Calculates correlation coefficients and p-values between all pairs of xvars and yvars.
            Rows with missing values are dropped pairwise. For each method, results are optionally
            saved as CSV and drawn with Plots.draw_corrmat().

            Args:
                - df (pd.DataFrame): wide-format dataframe where all elements of xvars and yvars are columns
                - xvars (list[str]): variables for the x-axis of the correlation matrix
                - yvars (list[str]): variables for the y-axis of the correlation matrix
                - methods (list[str]): correlation methods; elements must be pearson/spearman/kendall
                - kwargs (optional):
                    - dir_out, fname_out (str): if both given, matrices are saved as
                        f'{fname_out}_{method}_coeffs.csv' and f'{fname_out}_{method}_pvalues.csv'
                    - draw (bool): set to False to skip drawing the matrices
                    - any formatting argument of Plots.draw_corrmat()

            Returns:
                - df_coeffs (pd.DataFrame): correlation coefficients of the last method in methods
                - df_pvalues (pd.DataFrame): correlation p-values of the last method in methods
        '''

        assert isinstance(df, pd.DataFrame)
        for axis_vars in (xvars, yvars):
            assert isinstance(axis_vars, list)
            assert all(isinstance(var, str) for var in axis_vars)
            assert all(var in df.columns for var in axis_vars)

        corr_funcs = {
            'pearson': stats.pearsonr,
            'spearman': stats.spearmanr,
            'kendall': stats.kendalltau}

        for method in methods:
            # Rows are yvars, columns are xvars, hence the .at[var2, var1] indexing below
            df_coeffs = pd.DataFrame(columns=xvars, index=yvars)
            df_pvalues = pd.DataFrame(columns=xvars, index=yvars)

            for var1, var2 in itertools.product(xvars, yvars):
                df_pair = df[[var1, var2]].dropna()

                # A variable correlates perfectly with itself; avoids correlating a column with itself
                if var1 == var2:
                    df_coeffs.at[var2, var1] = 1
                    df_pvalues.at[var2, var1] = 0
                    continue

                result_corr = corr_funcs[method](df_pair[var1], df_pair[var2])
                df_coeffs.at[var2, var1] = round(result_corr.statistic, 3)
                df_pvalues.at[var2, var1] = round(result_corr.pvalue, 3)

            df_coeffs = df_coeffs.astype('float64')
            df_pvalues = df_pvalues.astype('float64')

            if 'dir_out' in kwargs and 'fname_out' in kwargs:
                df_coeffs.to_csv(os.path.join(kwargs['dir_out'], f"{kwargs['fname_out']}_{method}_coeffs.csv"))
                df_pvalues.to_csv(os.path.join(kwargs['dir_out'], f"{kwargs['fname_out']}_{method}_pvalues.csv"))

            if kwargs.get('draw') is not False:
                # n is the sample size of the last pair; it can differ between pairs when data are missing
                kwargs['corr_info'] = f'{method.upper()} correlation (n={df_pair.shape[0]})'
                kwargs['method'] = method
                Plots.draw_corrmat(df_coeffs=df_coeffs, df_pvalues=df_pvalues, **kwargs)

        return df_coeffs, df_pvalues

    @staticmethod
    def get_missing_scores(df_master: pd.DataFrame, measure_params: list[dict], **save) -> pd.DataFrame:
        ''' Get df of expected but missing scores. A score is missing if, for a participant and a
            timepoint in measure_param['tps'], there is not exactly one non-NaN score (at
            measure_param['time'] for measures with time).

            Args:
                - df_master (pd.DataFrame): long-form master df
                - measure_params (list[dict]): measure definitions, see config.py for the expected
                    format; uses keys 'measure', 'tps', 'has_time' and 'time' (if has_time is True)
                - save (dict; optional): keys dir_out and fname_out determine where output is saved

            Returns:
                - df_missing_scores (pd.DataFrame): columns pID, tp, measure, time
        '''

        df_missing_scores = pd.DataFrame(columns=['pID', 'tp', 'measure', 'time'])

        for measure_param in measure_params:
            has_time = measure_param['has_time']
            assert has_time in (True, False), f"Invalid has_time for {measure_param['measure']}: {has_time!r}"

            ### Every participant is expected to have exactly one score at each of the measure's tps
            for pID, tp in itertools.product(df_master.pID.unique(), measure_param['tps']):
                is_expected_row = (
                    (df_master.pID == pID) &
                    (df_master.tp == tp) &
                    (df_master.measure == measure_param['measure']))
                if has_time:
                    is_expected_row &= df_master.time == int(measure_param['time'])
                scores = df_master.loc[is_expected_row, 'score'].reset_index(drop=True)

                if len(scores) == 1 and not math.isnan(scores[0]):
                    continue

                df_missing_scores.loc[df_missing_scores.shape[0]] = {
                    'pID': pID,
                    'tp': tp,
                    'measure': measure_param['measure'],
                    'time': measure_param['time'] if has_time else math.nan}

        if save:
            df_missing_scores.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_missing_scores

    @staticmethod
    def get_dosedates(df_redcap: pd.DataFrame, df_trtalloc: pd.DataFrame, col_date: str = 'vrecord_date', **save) -> pd.DataFrame:
        ''' Get the dosing date and treatment condition of each participant.
            Dosing dates are taken from the A0 timepoint.

            Args:
                - df_redcap (pd.DataFrame): REDCap export with 'pID' and 'tp' columns
                - df_trtalloc (pd.DataFrame): treatment allocation with 'pID' and 'trt' columns
                - col_date (str): column of dosing dates in df_redcap
                - save (dict; optional): keys dir_out and fname_out determine where output is saved

            Returns:
                - df_dosedates (pd.DataFrame): columns pID, tp, date, cond
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(df_trtalloc, pd.DataFrame)

        df_dosedates = df_redcap.loc[df_redcap.tp == 'A0', ['pID', 'tp', col_date]]
        df_dosedates = df_dosedates.rename(columns={col_date: 'date'})
        df_dosedates['date'] = pd.to_datetime(df_dosedates['date'])

        df_dosedates = pd.merge(df_dosedates, df_trtalloc[['pID', 'trt']], on='pID')
        df_dosedates = df_dosedates.rename(columns={'trt': 'cond'})

        if save:
            df_dosedates.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_dosedates


class Plots:
    ''' Functions to help with figures '''

    @staticmethod
    def draw_translucent_boxplot(color, axis, alpha: float = 0.35, add_stripplot: bool = True, **kwargs) -> None:
        ''' Draws a translucent boxplot with an outline and an optional strip plot on top.

            Args:
                - color: color of the boxes, outlines and points
                - axis (matplotlib.axes.Axes): axis drawn on; its patches are made translucent
                - alpha (float): opacity of the box fill
                - add_stripplot (bool): if True, individual data points are drawn on top
                - kwargs: passed to sns.boxplot() and sns.stripplot(), e.g. data, x, y, ax
        '''

        ### Draw the boxes twice: filled boxes (made translucent below), then opaque outlines on top
        sns.boxplot(
            showfliers=False,
            color=color,
            **kwargs)

        sns.boxplot(
            fill=False,
            linewidth=0.85,
            showfliers=False,
            color=color,
            **kwargs)

        if add_stripplot:
            sns.stripplot(color=color, **kwargs)

        # Only the filled boxes are patches; lowering their alpha leaves outlines and points opaque
        for patch in axis.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, alpha))

    @staticmethod
    def draw_corrmat(df_coeffs: pd.DataFrame, df_pvalues: pd.DataFrame, **kwargs) -> None:
        ''' Draws a correlation matrix heatmap from the outputs of Analysis.get_corrmats().
            Cells are colored by coefficient and annotated with significance marks.

            Args:
                - df_coeffs (pd.DataFrame): correlation coefficients
                - df_pvalues (pd.DataFrame): correlation p-values
                - kwargs (optional):
                    - rotation (int): rotation of x tick labels; default 45
                    - title (str): figure title; defaults to kwargs['corr_info']
                    - xlabel, ylabel (str): axis labels
                    - dir_out, fname_out (str): if both given, the figure is saved; fname_out is
                        suffixed with f"_{kwargs['method']}" if method is given
        '''

        assert isinstance(df_coeffs, pd.DataFrame)
        assert isinstance(df_pvalues, pd.DataFrame)

        fig, ax = plt.subplots(dpi=300)

        # Annotations are significance marks (strings), hence fmt=''
        sns.heatmap(
            data=df_coeffs.astype(float),
            ax=ax,
            annot=df_pvalues.map(Helpers.sig_marking),
            vmin=-1,
            vmax=1,
            linewidths=.05,
            cmap='vlag',
            fmt='')

        plt.xticks(rotation=kwargs.get('rotation', 45))

        title = kwargs['title'] if 'title' in kwargs else kwargs['corr_info']
        ax.set_title(title, fontdict=commons_config.title_fontdict)

        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'], fontdict=commons_config.axislabel_fontdict)
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'], fontdict=commons_config.axislabel_fontdict)

        if 'dir_out' in kwargs and 'fname_out' in kwargs:
            fname_out = kwargs['fname_out']
            if 'method' in kwargs:
                fname_out += f"_{kwargs['method']}"

            Plots.save_fig(
                fig=fig,
                dir_out=kwargs['dir_out'],
                fname_out=fname_out,
                save_PNG=commons_config.save_PNG,
                save_SVG=commons_config.save_SVG)

    @staticmethod
    def draw_vitals(df_master: pd.DataFrame, measures: list[str] = ['dia', 'sys', 'hr'], **save) -> None:
        ''' Draw the in-dosing-session trajectory of vitals by condition, one figure per measure.

            Args:
                - df_master (pd.DataFrame): long-form master df with a 'condition' column (values C/T);
                    vitals are the rows with measure f'VITALS_{measure}'
                - measures (list[str]): vitals to draw; elements must be dia/sys/hr
                - save (dict; optional): keys dir_out and prefix_out determine where figures are
                    saved, as f'{prefix_out}_{measure}'
        '''

        assert isinstance(df_master, pd.DataFrame)

        ylabels = {
            'hr': 'Heart rate [BPM]',
            'dia': 'Diastolic BP [mmHg]',
            'sys': 'Systolic BP [mmHg]'}

        for measure in measures:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            df_master_measure = df_master.loc[df_master.measure == f'VITALS_{measure}']

            sns.lineplot(
                data=df_master_measure,
                ax=ax,
                x='time',
                y='score',
                hue='condition',
                markersize=10,
                legend=True,
                linewidth=2,
                markers=['o', 'D'],
                palette={
                    'C': '#56A0FB',
                    'T': '#F71480'},
                err_style='bars',
                err_kws={
                    'capsize': 4,
                    'elinewidth': 0.75,
                    'capthick': 0.75})

            ax.set_xlabel('Time [min]', fontdict=commons_config.axislabel_fontdict)
            ax.set_xticks([0, 30, 60, 90, 120, 240, 360, 420])
            ax.set_ylabel(ylabels[measure], fontdict=commons_config.axislabel_fontdict)

            ax.tick_params(axis='both', which='major', labelsize=commons_config.ticklabel_fontsize)
            sns.despine(top=True, right=True, left=False, bottom=False)
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            if save:
                Plots.save_fig(
                    fig=fig,
                    dir_out=save['dir_out'],
                    fname_out=f"{save['prefix_out']}_{measure}",
                    save_PNG=commons_config.save_PNG,
                    save_SVG=commons_config.save_SVG)

    @staticmethod
    def save_fig(fig, dir_out: str, fname_out: str, save_PNG: bool, save_SVG: bool) -> None:
        ''' Saves and then closes figure. dir_out is created if it does not exist.

            Args:
                - fig (matplotlib.figure.Figure): figure to save
                - dir_out (str): where to save fig
                - fname_out (str): file name without extension
                - save_PNG (bool): save fig as PNG?
                - save_SVG (bool): save fig as SVG?
        '''

        assert isinstance(save_PNG, bool)
        assert isinstance(save_SVG, bool)
        assert isinstance(dir_out, str)
        assert isinstance(fname_out, str)

        for fig_format, do_save in [('png', save_PNG), ('svg', save_SVG)]:
            if do_save:
                os.makedirs(dir_out, exist_ok=True)
                fig.savefig(
                    fname=os.path.join(dir_out, f'{fname_out}.{fig_format}'),
                    bbox_inches='tight',
                    format=fig_format,
                    dpi=300)

        plt.close(fig)


class CheckDf:
    ''' Check assumptions about long-form master dfs '''

    @staticmethod
    def check_master(df_master: pd.DataFrame, measure_params: list[dict], timings: list[str] = commons_config.timings, conditions: list[str] = ['C', 'T']) -> pd.DataFrame:
        ''' Run all checks on df_master. Failed checks raise an AssertionError.

            Args:
                - df_master (pd.DataFrame): long-form master df
                - measure_params (list[dict]): measure definitions, see config.py for the expected format
                - timings (list[str]): allowed values of the 'timing' column
                - conditions (list[str]): allowed values of the 'condition' column

            Returns:
                - df_duplicate_rows (pd.DataFrame): duplicate rows, see check_duplicate_rows()
        '''

        df_duplicate_rows = CheckDf.check_duplicate_rows(df_master)
        CheckDf.check_conditions(df_master, conditions)
        CheckDf.check_time(df_master, measure_params)
        CheckDf.check_timings(df_master, timings)

        return df_duplicate_rows

    @staticmethod
    def check_duplicate_rows(df_master: pd.DataFrame, cols: list[str] = commons_config.cols_checkduplicates) -> pd.DataFrame:
        ''' Returns all rows (restricted to cols) that share their values of cols with another row '''

        df = df_master[cols]
        return df[df.duplicated(keep=False)]

    @staticmethod
    def check_conditions(df_master: pd.DataFrame, conditions: list[str]) -> None:
        ''' Check that every value of the 'condition' column is one of conditions '''

        assert df_master.condition.isin(conditions).all()

    @staticmethod
    def check_time(df_master: pd.DataFrame, measure_params: list[dict]) -> None:
        ''' Check the 'time' column of each measure based on its has_time flag:
            - has_time=True: every row of the measure must have a numeric (non-NaN) time
            - has_time=False: every row of the measure must have time=NaN
        '''

        for measure_param in measure_params:
            measure = measure_param['measure']
            times = df_master.loc[df_master.measure == measure, 'time']
            if times.empty:
                continue

            if measure_param['has_time'] is True:
                assert pd.to_numeric(times, errors='coerce').notna().all(), (
                    f"Measure '{measure}' has has_time=True but some rows have non-numeric/NaN time.")
            elif measure_param['has_time'] is False:
                assert times.isna().all(), (
                    f"Measure '{measure}' has has_time=False but some rows have a non-NaN time.")
            else:
                assert False, f"Measure '{measure}' has invalid has_time value: {measure_param['has_time']!r}"

    @staticmethod
    def check_timings(df_master: pd.DataFrame, timings: list[str] = commons_config.timings) -> None:
        ''' Check that every value of the 'timing' column is one of timings. Default timings are:
            - bsl: measured only at baseline, e.g. demographic variables
            - change: measured both before and after treatment, e.g. depression scores
            - in_dose: measured during the dosing session, thus have a value for time, e.g. vitals
            - post_dose: measured only post-dosing, e.g. measures of trip quality like MEQ
            - post_trt: measured only post-treatment, e.g. treatment satisfaction
        '''

        assert isinstance(df_master, pd.DataFrame)
        assert df_master.timing.isin(timings).all()


class Helpers:
    ''' Various helper functions '''

    @staticmethod
    def sig_marking(value: float) -> str:
        ''' Converts a p-value to standard significance marks: *** <0.001, ** <0.01, * <0.05, otherwise '' '''

        if value < 0.001:
            return '***'
        elif value < 0.01:
            return '**'
        elif value < 0.05:
            return '*'
        else:
            return ''

    @staticmethod
    def get_col_sum_score(measure_param: dict) -> str:
        ''' Name of the REDCap column where calc_sum_scores() stores the sum score of a measure '''

        return f"{measure_param['measure']}_LOCAL_SCORE"

    @staticmethod
    def has_time(df: pd.DataFrame, measure: str) -> bool:
        ''' Detects whether the given measure has time, i.e. whether it is measured at multiple
            times within a timepoint (all its rows have a numeric time) or not (all rows have time=NaN).

            Args:
                - df (pd.DataFrame): long-form master df
                - measure (str): name of measure

            Returns:
                - has_time (bool): True if the measure has time

            Raises:
                - UndecidedHasTime: if the measure has no rows, or has both NaN and numeric times
        '''

        times = df.loc[df.measure == measure, 'time']
        is_all_nan = times.isna().all()
        is_all_number = times.notna().all()

        # Both are True if there are no rows, both are False if NaN and numeric times are mixed
        if is_all_nan == is_all_number:
            raise UndecidedHasTime(measure)

        return bool(is_all_number)


class UndecidedHasTime(Exception):
    ''' Raised when it can not be decided whether a measure has time, see Helpers.has_time() '''

    def __init__(self, measure):
        self.measure = measure
        super().__init__(self.measure)


class IncompleteScore(Exception):
    ''' Raised when a score can not be calculated due to incomplete data; df_exception holds the offending rows '''

    def __init__(self, msg, df_exception):
        self.msg = msg
        self.df_exception = df_exception
        super().__init__(self.msg, self.df_exception)
