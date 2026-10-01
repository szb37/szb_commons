import commons_codebase.src.eqscores_hamd as eqscores_hamd
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DataWrangl():
    ''' Functions for common data wrangling tasks '''

    @staticmethod
    def clean_df_redcap(df_redcap, rename_cols={}, rename_tps={}, rm_tps=[],):
        """
        Rename_columns, rename timepoints, remove spurious timepoints.

        Args:
            df_redcap (pd.DataFrame): Raw REDCap export dataframe.
            rename_tps (dict, optional): Dictionary mapping old to new timepoint names (e.g., {'tp_old': 'tp_new'}). Default is {} (no renaming).
            rm_tps (list, optional): List of timepoints to remove from the dataframe. Rows with tp in this list will be dropped. Default is [].

        Returns:
            pd.DataFrame: Cleaned REDCap dataframe.
        """

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(rename_cols, dict)
        assert isinstance(rename_tps, dict)
        assert isinstance(rm_tps, list)

        ### Rename columns
        df_redcap = df_redcap.rename(columns = rename_cols)
        ### Rename timepoints
        df_redcap['tp'] = df_redcap['tp'].replace(rename_tps)
        ### Remove spurious timepoints
        df_redcap = df_redcap.loc[~df_redcap.tp.isin(rm_tps)]

        ### Check if all tps have been renamed or not
        valid_tps = list(set(rename_tps.values()))
        invalid_tps = list(set(df_redcap.loc[~df_redcap.tp.isin(valid_tps), 'tp']))
        if len(invalid_tps)>0:
            print(f'Non-defined tps found in df_redcap: {invalid_tps}')

        return df_redcap

    @staticmethod
    def clean_pIDs(df_redcap, n_prefix_chars=0,):
        ### Convert pIDs to numeric
        ### Remove first n_prefix_chars from pID prior to INT conversion
        #   For example, if pID is 'PDP2001', and n_prefix_chars=3, then result is 2001
        if n_prefix_chars > 0:
            df_redcap['pID'] = df_redcap['pID'].astype(str).str[n_prefix_chars:]

        ### Print & delete non-convertible pIDs
        pID_numeric = pd.to_numeric(df_redcap['pID'], errors='coerce')
        invalid_pids = df_redcap.loc[pID_numeric.isna(), 'pID'].unique()
        if len(invalid_pids) > 0:
            print(f'Unique pIDs that could not be converted to int (deleting rows):')
            for invalid_pid in invalid_pids:
                print(f'\t{invalid_pid}')

        df_redcap = df_redcap.loc[~pID_numeric.isna()].copy()
        df_redcap['pID'] = pID_numeric[~pID_numeric.isna()].astype(int)

        return df_redcap

    @staticmethod
    def replace_tps(df, repeated_tps={}):
        ''' Replace repeated timepoints in a long-form dataframe.

            For each (new_tp, old_tp) pair in `repeated_tps`, rows tagged with
            `new_tp` override rows with `old_tp` for the same (pID, measure).
            Discrepancies in non-value identifying columns raise a warning.
            After replacement, any residual duplicated (pID, tp, measure)
            combinations are flagged and returned separately.

            Args:
                - df (pd.DataFrame): long-form dataframe with columns
                    including 'pID', 'tp', 'measure', 'score' (and optionally
                    'delta_score').
                - repeated_tps (dict): mapping of {new_tp: old_tp} indicating
                    which repeat-timepoint rows should overwrite which
                    originals (e.g. {'bsl_R1': 'bsl', 'A0_R1': 'A0'}).

            Returns:
                - df (pd.DataFrame): dataframe after replacement, with the
                    new_tp rows removed.
                - df_replaced_rows (pd.DataFrame): all rows (with every
                    column, pre-replacement state) whose values were
                    overwritten by a repeat-timepoint row. Empty if none.
                - df_duplicate_rows (pd.DataFrame): all rows (with every
                    column) that share a (pID, tp, measure) combination
                    appearing more than once after replacement. Empty if
                    none exist.
        '''

        key_cols = ['pID', 'measure', 'time']
        ignore_cols = set(key_cols + ['tp', 'score', 'delta_score'])
        compare_cols = [c for c in df.columns if c not in ignore_cols]

        replaced_rows = []

        for new_tp, old_tp in repeated_tps.items():
            new_rows = df.loc[df.tp == new_tp]
            for new_idx, new_row in new_rows.iterrows():
                pID = new_row['pID']
                measure = new_row['measure']
                old_mask = (
                    (df.pID == pID) &
                    (df.measure == measure) &
                    (df.tp == old_tp))
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
                            f"Discrepancy for pID={pID}, measure={measure}, column '{col}': "
                            f"{new_tp}={new_val!r} vs {old_tp}={old_val!r}")

                ### Capture the pre-replacement row before overwriting
                replaced_rows.append(old_row.copy())

                ### Replace old row's values with new row's (keep tp=old_tp)
                replacement = new_row.copy()
                replacement['tp'] = old_tp
                df.loc[old_matches.index[0], df.columns] = replacement.values

            ### Drop the repeat rows now that their data has been merged into old_tp
            df = df.loc[df.tp != new_tp].reset_index(drop=True)

        ### Build df of replaced rows (pre-replacement state)
        if replaced_rows:
            df_replaced_rows = pd.DataFrame(replaced_rows).reset_index(drop=True)
            df_replaced_rows = df_replaced_rows.sort_values(['pID', 'measure', 'tp']).reset_index(drop=True)
        else:
            df_replaced_rows = pd.DataFrame(columns=df.columns)

        ### Detect residual duplicates on (pID, tp, measure)
        dup_mask = df.duplicated(subset=['pID', 'tp', 'measure', 'time'], keep=False)
        df_duplicate_rows = df.loc[dup_mask].sort_values(['pID', 'measure', 'tp']).reset_index(drop=True)

        return df, df_replaced_rows, df_duplicate_rows

    @staticmethod
    def get_df_tp_ndays(df_redcap:pd.DataFrame, col_date:str, **save) -> pd.DataFrame:
        ''' Get df of the median number of days since baseline for each timepoint.

            Args:
                - df_redcap (pd.DataFrame): raw REDCap export df
                - col_date (str): column of dates in REDCap df
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df_tp_ndays: df of avg days since baseline (to a given tp)
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(col_date, str)

        df = df_redcap.rename(columns={col_date: 'date',})
        df = df.loc[(df.study_visit_completion_record_complete==2)]

        df = df[['pID', 'tp', 'date']]
        df = df.dropna()
        df.date = pd.to_datetime(df.date)
        df.reset_index(drop=True, inplace=True)
        cidx_date = df.columns.get_loc('date')

        ### Prep outpur
        df_tp_ndays_rows = []
        df_tp_ndays_rows.append(['bsl', 0, math.nan])

        ### Iterate through all patients / tps
        for tp in df.tp.unique():

            if tp=='bsl':
                continue
            ndays=[]

            for pID in df.pID.unique():
                row_bsl = df.loc[(df.pID==pID) & (df.tp=='bsl')]
                row_tp = df.loc[(df.pID==pID) & (df.tp==tp)]

                # calculate tp-bsl in days
                # ignore if there is not exactly 1 row for either tp or bsl
                if (row_bsl.shape[0]==1) & (row_tp.shape[0]==1):
                    ndays.append(
                        (df.iloc[row_tp.index[0], cidx_date] - df.iloc[row_bsl.index[0], cidx_date]).days)

            ### Create row with median number of days between baseline and tp
            df_tp_ndays_rows.append([
                tp,
                round(np.median(np.array(ndays))),
                len(ndays)])

        ### Convert list of rows to df
        df_tp_ndays = pd.DataFrame(columns=['tp', 'ndays', 'n'], data=df_tp_ndays_rows)
        df_tp_ndays.sort_values(by='ndays', inplace=True)

        ### Save if needed, return output
        if save!={}:
            df_tp_ndays.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_tp_ndays

    @staticmethod
    def get_df_measure(df_redcap:pd.DataFrame, measure_param:dict, cols_to_keep:list[str]=commons_config.cols_to_keep, **save)-> pd.DataFrame:
        ''' Returns all completed scores of a given measure in long-formatted df.
            Rows are removed that have missing value in any of the pID/tp/score columns.

            Args:
                - df_redcap: raw export from REDCap
                - cols_to_keep: what columns from the REDCAP df should be kept in returned df
                - measure_param (dict): dictionary that defines the measure, see config.py for example dict. Need to have keys:
                        - 'instrument': value of the "instrument" column in the returned df;
                        - 'measure': value of the "measure" column in the returned df; "measure" is typcially name of the scale or a subscale
                        - 'col_complete': column name in df_redcap, which tracks if row was completed; if set to None, completion is not checked, if a str is provided only rows are kept where its value is 2
                        - 'col_score': column name in df_redcap, which stores score
                        - 'type': value of the "type" column in the returned df; usefull to distinguish structure of measures
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df (pd.DataFrame): longform df with all scores of the defined measure
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(measure_param, dict)
        assert all([field in measure_param.keys() for field in
            ['instrument', 'measure', 'timing', 'col_complete', 'col_score',]])
        assert isinstance(cols_to_keep, list)

        # define variables
        df = df_redcap.copy()
        col_score = measure_param['col_score']
        col_complete = measure_param['col_complete']

        # reduce dataframe
        df = df.loc[(df[col_complete]==2)]
        df = df[cols_to_keep + [col_score]]

        # rename / add bookkeeping columns
        df.rename(columns={col_score: 'score'}, inplace=True)
        df['instrument'] = measure_param['instrument']
        df['measure'] = measure_param['measure']
        df['timing'] = measure_param['timing']

        # deal with time
        if 'time' in measure_param.keys():
            df['time'] = measure_param['time']
        else:
            df['time'] = math.nan

        ### Cleanup, save, return output
        df = df.dropna(subset=(['pID', 'tp', 'score']))
        df['score'] = df['score'].astype('float64')
        df['time'] = df['time'].astype('float64')
        df = df[cols_to_keep+['timing','instrument','measure','time','score']]
        df.reset_index(inplace=True, drop=True)

        if save!={}:
            df.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df

    @staticmethod
    def calc_sum_scores(df_redcap:pd.DataFrame, measure_param:dict) -> pd.DataFrame:
        ''' Calculates the scores for measures defined by measure_param, see config for formatting.

            Args:
                - df_redcap (pd.DataFrame): raw export from REDCap
                - measure_param (dict): dictionary that defines the measure, see config.py for example dict. Need to have keys:
                    - 'instrument': value of the "instrument" column in the returned df;
                    - 'measure': value of the "measure" column in the returned df; "measure" is typcially name of the scale or a subscale
                    - 'col_complete': column name in df_redcap, which tracks if row was completed; if set to None, completion is not checked, if a str is provided only rows are kept where its value is 2
                    - 'col_score': column name in df_redcap, which stores score
                    - 'type': value of the "type" column in the returned df; usefull to distinguish structure of measures

            Returns:
                - df_redcap (pd.DataFrame): REDCap df with scores added
        '''

        col_complete = measure_param['col_complete']
        sum_score = measure_param['sum_score']
        col_items = sum_score['col_items']
        reverse_items = sum_score['reverse_items']
        min_score = sum_score['min_score']
        max_score = sum_score['max_score']
        col_score = measure_param['col_score']

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(col_items, list)
        assert all([col_item in df_redcap.columns for col_item in col_items])
        assert col_complete in df_redcap.columns
        assert col_score not in df_redcap.columns, f"'col_score' (={measure_param['col_score']}) already deifned in REDCap df"

        ### Reverse items if there is any
        if len(reverse_items)!=0:
            original_scores = df_redcap.loc[(df_redcap[col_complete]==2), reverse_items]
            reverse_scores = \
                (max_score + min_score) - df_redcap.loc[(df_redcap[col_complete]==2), reverse_items]
            df_redcap.loc[(df_redcap[col_complete]==2), reverse_items] = reverse_scores

        ### Calculate sum scores
        df_redcap.loc[(df_redcap[col_complete]==2), col_score] = df_redcap.loc[(df_redcap[col_complete]==2), col_items].sum(axis=1)

        ### Reverse back reverse-items back to original as the column may get processed again in different contexts
        if len(reverse_items)!=0:
            df_redcap.loc[(df_redcap[col_complete]==2), reverse_items] = original_scores

        ### Check missing items
        rows_df_missingitems = []
        for row in df_redcap.loc[(df_redcap[col_complete]==2)].itertuples():

            cols_missing=[]
            for col in col_items:
                assert (eval(f'row.{col}') is not None)
                if (math.isnan(eval(f'row.{col}'))):
                    cols_missing.append(col)

            if cols_missing!=[]:
                rows_df_missingitems.append([row.pID, row.tp, measure_param['measure'], cols_missing])
                df_redcap.loc[row.Index, col_score] = math.nan

        ### Return
        df_missingitems = pd.DataFrame(columns=['pID', 'tp', 'measure', 'missing_items'], data=rows_df_missingitems)
        return df_redcap, df_missingitems

    @staticmethod
    def add_delta_scores(df_master:pd.DataFrame, delta_from_tp:str='bsl', delta_from_time:int=0) -> pd.DataFrame:
        ''' Add delta_score column to a longform trial dataframe.

            For each measure, the baseline reference is chosen once (not per row):
                - If the measure has NO time column populated (all rows have
                  time=NaN for that measure), baseline is the row at
                  tp == delta_from_tp for the same (pID, measure). delta_score
                  is score - baseline_score for all rows of that measure.
                - If the measure has time (all rows have a non-NaN numeric time
                  for that measure), baseline is the row at
                  time == delta_from_time for the same (pID, measure, tp).
                  delta_score is score - baseline_score for all rows at that tp.

            Vectorized via groupwise merges rather than per-row iteration.

            Args:
                - df_master (pd.DataFrame): longform df of trial data
                - delta_from_tp (str): value in 'tp' that designates baseline
                    for measures without time.
                - delta_from_time (int): value in 'time' that designates
                    baseline for measures with time.

            Returns:
                - df_master (pd.DataFrame): input df with 'delta_score' column added
        '''

        assert isinstance(df_master, pd.DataFrame)
        assert isinstance(delta_from_tp, str)
        assert isinstance(delta_from_time, int)

        df_master = df_master.copy()
        df_master['delta_score'] = math.nan

        undecided_has_time = []
        pieces = []

        for measure, sub in df_master.groupby('measure', sort=False):
            try:
                has_time = Helpers.has_time(df_master, measure)
            except UndecidedHasTime as e:
                undecided_has_time.append(e.measure)
                pieces.append(sub)
                continue

            if has_time:
                bsl = (sub.loc[sub['time'] == delta_from_time, ['pID', 'tp', 'score']]
                          .rename(columns={'score': '_bsl_score'}))
                merged = sub.merge(bsl, on=['pID', 'tp'], how='left')
            else:
                bsl = (sub.loc[sub['tp'] == delta_from_tp, ['pID', 'score']]
                          .rename(columns={'score': '_bsl_score'}))
                merged = sub.merge(bsl, on=['pID'], how='left')

            merged.index = sub.index
            merged['delta_score'] = merged['score'] - merged['_bsl_score']
            pieces.append(merged.drop(columns=['_bsl_score']))

        df_master = pd.concat(pieces).sort_index()

        if undecided_has_time:
            undecided_has_time = set(undecided_has_time)
            print(f"Can not decide whether measure has time: {[measure for measure in undecided_has_time]}")

        return df_master

    @staticmethod
    def get_df_vitals(df_redcap:pd.DataFrame, measure_param:dict, **save) -> pd.DataFrame:
        ''' Special case of get_df_measure() to deal with the idiosyncrasies of vitals measures.
            Specifically, there is either 1 or 2 readings of vitals.
            If there are 2 readings, then the avg is propagated to the output.

            Args:
                - df_redcap (pd.DataFrame): raw REDCap export df
                - measure_param (dict): dictionary that defines the measure, see config.py for example dict. Need to have keys:
                        - 'instrument': value of the "instrument" column in the returned df;
                        - 'measure': value of the "measure" column in the returned df; "measure" is typcially name of the scale or a subscale
                        - 'col_complete': column name in df_redcap, which tracks if row was completed; if set to None, completion is not checked, if a str is provided only rows are kept where its value is 2
                        - 'col_score': column name in df_redcap, which stores score
                        - 'type': value of the "type" column in the returned df; usefull to distinguish structure of measures
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df_vitals (pd.DataFrame): long-form dataframe of vitals data
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(measure_param, dict)
        assert all([field in measure_param.keys() for field in
            ['instrument', 'measure', 'timing', 'col_complete', 'col_score',]])

        df = df_redcap.loc[(df_redcap[measure_param['col_complete']]==2)]
        rows_vitals = []

        for row in df.itertuples():
            pID = row.pID
            tp = row.tp

            score1 = eval(f'row.{measure_param['col_score']}_1') # double check if the format matches with REDCap survey, lab is inconsistent across trials
            score2 = eval(f'row.{measure_param['col_score']}_2')

            if (not math.isnan(score1)) and (math.isnan(score2)):
                score = score1 # no second measure was taken
            elif (math.isnan(score1)) and (not math.isnan(score2)):
                assert False   # second measure should only exist if there was a first measure
            elif (not math.isnan(score1)) and (not math.isnan(score2)):
                score = round((score1+score2)/2, 3)
            else:
                continue # no measure taken

            rows_vitals.append([
                pID,
                tp,
                'in-dose',
                measure_param['instrument'],
                measure_param['measure'],
                measure_param['time'],
                score,])

        df_vitals = pd.DataFrame(columns=[
            'pID',
            'tp',
            'timing',
            'instrument',
            'measure',
            'time',
            'score',], data=rows_vitals)

        ### Cleanup, save, return output
        df_vitals.dropna(subset=['pID', 'tp', 'score'], inplace=True)
        df_vitals['score'] = df_vitals['score'].astype('float64')
        df_vitals['time'] = df_vitals['time'].astype('float64')
        df_vitals = df_vitals[['pID','tp','timing','instrument','measure','time','score']]
        df_vitals.reset_index(inplace=True, drop=True)

        if save!={}:
            df_vitals.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_vitals

    @staticmethod
    def widen_master(df_master:pd.DataFrame, xvars:list[str], x_tp:str, x_use_delta:bool, yvars:list[str], y_tp:str, y_use_delta:bool) -> pd.DataFrame:
        ''' Convert long-form master df to wide-format df for correlation analysis

            Args:
                - df_master(pd.DataFrame): master df of the trial
                - xvars(list[str]): list of measures, i.e. one set of column headers in the resulting wide-format df
                - x_tp(str): use scores from what timepoint for measures in the xvars list
                - x_use_delta(bool): use delta_score/score in the measure's column for measures in the xvars list
                - yvars(list[str]): list of measures, i.e. one set of column headers in the resulting wide-format df
                - y_tp(str): use scores from what timepoint for measures in the yvars list
                - y_use_delta(bool): use delta_score/score in the measure's column for measures in the xvars list

            Return:
                - df(pd.DataFrame): wide-format data frame
        '''

        assert isinstance(df_master, pd.DataFrame)
        for axis in ['x', 'y']:
            assert isinstance(eval(f'{axis}vars'), list)
            assert sum([isinstance(measure, str) for measure in eval(f'{axis}vars')])
            assert isinstance(eval(f'{axis}_tp'), str)

        df = df_master.loc[
            ((df_master.tp==x_tp) & (df_master.measure.isin(xvars))) |
            ((df_master.tp==y_tp) & (df_master.measure.isin(yvars)))]

        if x_use_delta:
            df.loc[(df.tp==x_tp) & (df.measure.isin(xvars)), 'score'] = df.loc[(df.tp==x_tp) & (df.measure.isin(xvars)), 'delta_score']

        if y_use_delta:
            df.loc[(df.tp==y_tp) & (df.measure.isin(yvars)), 'score'] = df.loc[(df.tp==y_tp) & (df.measure.isin(yvars)), 'delta_score']

        df = pd.pivot_table(df, index=['pID',], columns='measure', values='score', dropna=False)
        df.reset_index(inplace=True)

        return df


class Analysis():
    ''' Functions for data analysis '''

    @staticmethod
    def get_df_observed(df_master:pd.DataFrame, digits:int=3, **save) -> pd.DataFrame:
        ''' Creates a dataframe with the observed mean and SD of all measures at every tp.
            Missing data are ignored from the mean/sd calculations.

            Args:
                - df_master (pd.DataFrame): long-form master dataframe containing all data
                - digits (int): round mean and SD to how many digits?
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df_observed: df of observed means and SDs at every tp
        '''

        ### Initate output
        measures = df_master.measure.unique().tolist()
        tps = df_master.tp.unique().tolist()
        rows_observed = []

        ### Iterate through measures and tps, calculate means and SDs
        for measure, tp in itertools.product(measures, tps):

            scores = df_master.loc[(df_master.measure==measure) & (df_master.tp==tp)].score.dropna().tolist()
            if len(scores)<3: # skip if not enough data to calc SD
                continue

            rows_observed.append([
                measure,
                tp,
                len(scores),
                round(mean(scores), digits),
                round(stdev(scores), digits),])

        ### Create DF from list of rows
        df_observed = pd.DataFrame(
            columns=['measure','tp','n','mean','sd'],
            data=rows_observed)

        ### Save if needed, return output
        if save!={}:
            df_observed.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_observed

    @staticmethod
    def get_corrmats(df:pd.DataFrame, xvars:list[str], yvars:list[str], methods:list[str]=commons_config.corr_methods, **kwargs):
        ''' Calculates and corr coeffs and associated p-values between all pairs of xvars and yvars
            Correlations are calculated with 'pearson', 'spearman' and 'kendall' methods

            Args:
                - df (pd.DataFrame): wide-format dataframe where all elements of xvars and yvars are columns
                - xvars (list of strs): variables for the x-axis of corr matrix
                - yvars (list of strs): variables for the y-axis of corr matrix
                - methods (list of strs): what correlation method to use; elements must be pearson/spearman/kendall
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df_coeffs (pd.DataFrame): dataframe of correlation coefficients
                -df_pvalues (pd.DataFrame): dataframe of correlation p-values
        '''

        assert isinstance(df, pd.DataFrame)
        for axis in ['x','y']:
            assert isinstance(eval(f'{axis}vars'), list)
            assert sum([isinstance(var, str) for var in eval(f'{axis}vars')])
            assert sum([var in df.columns for var in eval(f'{axis}vars')])

        df_coeffs = pd.DataFrame(columns=xvars, index=yvars)
        df_pvalues = pd.DataFrame(columns=xvars, index=yvars)

        for method in methods:
            for var1, var2 in itertools.product(xvars, yvars):

                df_pair = df[[var1, var2]]
                df_pair = df_pair.dropna()

                if var1 == var2:
                    df_coeffs.at[var2, var1] = 1
                    df_pvalues.at[var2, var1] = 0
                    continue

                if method == 'pearson':
                     result_corr = stats.pearsonr(df_pair[var1], df_pair[var2])
                elif method == 'spearman':
                     result_corr = stats.spearmanr(df_pair[var1], df_pair[var2])
                elif method == 'kendall':
                     result_corr = stats.kendalltau(df_pair[var1], df_pair[var2])

                df_coeffs.at[var2, var1] = round(result_corr.statistic, 3)
                df_pvalues.at[var2, var1] = round(result_corr.pvalue, 3)

            ### Clean up & save/visualize
            # Convert to numeric
            df_coeffs = df_coeffs.astype('float64')
            df_pvalues = df_pvalues.astype('float64')

            # Save results if needed
            if (('dir_out' in kwargs) and ('fname_out' in kwargs)):
                df_coeffs.to_csv(os.path.join(kwargs['dir_out'], f'{kwargs['fname_out']}_{method}_coeffs.csv'))
                df_pvalues.to_csv(os.path.join(kwargs['dir_out'], f'{kwargs['fname_out']}_{method}_pvalues.csv'))

            # Draw correlation unless draw is explicitly False
            if ('draw' in kwargs):
                if (kwargs['draw'] is False):
                    return df_coeffs, df_pvalues

            kwargs['corr_info'] = f'{method.upper()} correlation (n={df_pair.shape[0]})'
            kwargs['method'] = method

            Plots.draw_corrmat(
                df_coeffs=df_coeffs,
                df_pvalues=df_pvalues,
                **kwargs,)

    @staticmethod 
    def get_missing_scores(df_master, measure_params, **save):
        ''' Get df of missing scores
        Args:
            - df_master (pd.DataFrame): long-form master df
            - measure_params (list of dict): list of measure parameters, see config.py for format
            - save (optional, dict): where to save results if want to

        Returns:
            - df_res (pd.DataFrame): df of missing scores
        '''

        df_missing_scores = pd.DataFrame(columns=['pID', 'tp', 'measure', 'time'])

        for measure_param in measure_params:

            if measure_param['has_time'] is True:
                for pID, tp in itertools.product(df_master.pID.unique(), measure_param['tps'],):
                    scores = df_master.loc[
                        (df_master.pID==pID) & 
                        (df_master.tp==tp) & 
                        (df_master.time==int(measure_param['time'])) & 
                        (df_master.measure==measure_param['measure'])].score.reset_index(drop=True)
                    
                    if ((len(scores)==1) and (not math.isnan(scores[0]))):
                        continue
                    else:
                        df_missing_scores.loc[df_missing_scores.shape[0]] = {
                            'pID': pID,
                            'tp': tp,
                            'measure': measure_param['measure'],
                            'time': measure_param['time']}

            elif measure_param['has_time'] is False:

                for pID, tp in itertools.product(df_master.pID.unique(), measure_param['tps']):

                    scores = df_master.loc[
                        (df_master.pID==pID) & 
                        (df_master.tp==tp) & 
                        (df_master.measure==measure_param['measure'])].score.reset_index(drop=True)


                    if ((len(scores)==1) and (not math.isnan(scores[0]))):
                        continue
                    else:
                        df_missing_scores.loc[df_missing_scores.shape[0]] = {
                            'pID': pID,
                            'tp': tp,
                            'measure': measure_param['measure'],
                            'time': math.nan}

        return df_missing_scores

    @staticmethod
    def get_dosedates(df_redcap, df_trtalloc, col_date='vrecord_date', **save):
        """ Add tp/condition column to REDCap export of AEs

            Args:
                - df_redcap (pd.DataFram): REDCAP export
                - df_ae (pd.DataFram): REDCAP export of AEs

            Returns:
                - df_ae (pd.DataFrame):
        """

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(df_trtalloc, pd.DataFrame)

        df_dosedates = df_redcap.loc[(df_redcap.tp=='A0')]

        df_dosedates = df_dosedates[['pID', 'tp', col_date]]
        df_dosedates = df_dosedates.rename(columns={col_date: 'date',})
        df_dosedates['date'] = pd.to_datetime(df_dosedates['date'])

        df_dosedates = pd.merge(df_dosedates, df_trtalloc[['pID', 'trt']], on='pID')
        df_dosedates = df_dosedates.rename(columns={'trt':'cond'})

        if save!={}:
            df_dosedates.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_dosedates


class Plots():
    ''' Functions to help with figures '''

    @staticmethod
    def draw_translucent_boxplot(color, axis, alpha=0.35, add_stripplot=True, **kwargs):
        ''' Draws on axis a translucent boxplot with optional strip plot on top.
            Shortcut to achieve the look I like.
        '''

        sns.boxplot(
            showfliers=False,
            color=color,
            **kwargs,)

        sns.boxplot(
            fill=False,
            linewidth=0.85,
            showfliers=False,
            color=color,
            **kwargs,)

        if add_stripplot:
            sns.stripplot(color=color, **kwargs,)

        for patch in axis.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, alpha))

    @staticmethod
    def draw_corrmat(df_coeffs, df_pvalues, **kwargs):
        ''' Draws correlation matrix heatmap using outputs of get_corrmat()

            Args:
                df_coeffs (pd.DataFrame): dataframe of correlation coefficients
                df_pvalues (pd.DataFrame): dataframe of correlation p-values
                kwargs (dict, optional): optinal info to format & save fig
        '''

        assert isinstance(df_coeffs, pd.DataFrame)
        assert isinstance(df_pvalues, pd.DataFrame)

        fig, ax = plt.subplots(dpi=300)

        sns.heatmap(
            data = df_coeffs.astype(float),
            ax = ax,
            annot = df_pvalues.applymap(Helpers.sig_marking),
            vmin = -1,
            vmax = 1,
            linewidths = .05,
            cmap = 'vlag',
            fmt = '')

        if 'rotation' in kwargs:
            rotation = kwargs['rotation']
        else:
            rotation = 45
        plt.xticks(rotation=rotation)

        if 'title' in kwargs:
            ax.set_title(kwargs['title'], fontdict=commons_config.title_fontdict)
        else:
            ax.set_title(kwargs['corr_info'], fontdict=commons_config.title_fontdict)

        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'], fontdict=commons_config.axislabel_fontdict)

        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'], fontdict=commons_config.axislabel_fontdict)

        if (('dir_out' in kwargs) and ('fname_out' in kwargs)):

            if 'method' in kwargs:
                kwargs['fname_out'] = kwargs['fname_out']+f'_{kwargs['method']}'

            Plots.save_fig(
                fig = fig,
                dir_out  = kwargs['dir_out'],
                fname_out = kwargs['fname_out'],
                save_PNG = commons_config.save_PNG,
                save_SVG = commons_config.save_SVG,)

    @staticmethod
    def draw_vitals(df_master, measures=['dia', 'sys', 'hr'], **save):
        ''' Draw in-dosing-session trajectory of vitals

            Args:
                df_master (pd.DataFrame): long-form master df
                measures (list): list of vitals;
                dir_out (str): where to save results
                save (bool): save figure?
        '''

        assert isinstance(df_master, pd.DataFrame)
        assert isinstance(dir_out, str)
        assert isinstance(prefix_out, str)
        assert isinstance(save, bool)

        for measure in measures:

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            df_master_measure = df_master.loc[(df_master.measure==f'VITALS_{measure}')]

            ax = sns.lineplot(
                data = df_master_measure,
                x = 'time',
                y = 'score',
                #hue = 'tp',
                hue = 'condition',
                markersize = 10,
                legend = True,
                linewidth=2,
                markers = [
                    "o", "D"],
                palette = {
                    'C': '#56A0FB',
                    'T': '#F71480'},
                err_style = "bars",
                err_kws={
                    'capsize': 4,
                    'elinewidth': 0.75,
                    'capthick': 0.75},)

            ax.set_xlabel('Time [min]', fontdict=commons_config.axislabel_fontdict)
            ax.set_xticks([0, 30, 60, 90, 120, 240, 360, 420])

            if measure=='hr':
                ax.set_ylabel('Heart rate [BPM]', fontdict=commons_config.axislabel_fontdict)
            elif measure=='dia':
                ax.set_ylabel('Diastolic BP [mmHg]', fontdict=commons_config.axislabel_fontdict)
            elif measure=='sys':
                ax.set_ylabel('Systolic BP [mmHg]', fontdict=commons_config.axislabel_fontdict)
            else:
                assert False

            ax.tick_params(axis='both', which='major', labelsize=commons_config.ticklabel_fontsize)
            sns.despine(top=True, right=True, left=False, bottom=False)
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)

            if save!={}:
                Plots.save_fig(
                    fig = fig,
                    dir_out = dir_out,
                    fname_out = f'{save['prefix_out']}_{measure}',
                    save_PNG=commons_config.savePNG,
                    save_SVG=commons_config.saveSVG)

    @staticmethod
    def save_fig(fig, dir_out, fname_out, save_PNG, save_SVG):
        ''' Saves and then closes figure

            Args:
                - save_PNG (bool): save fig as PNG?
                - save_SVG (bool): save fig as SVG?
                - dir_out (str): where to save fig
                - fname_out (str): name of file saves
        '''

        assert isinstance(save_PNG, bool)
        assert isinstance(save_SVG, bool)
        assert isinstance(dir_out, str)
        assert isinstance(fname_out, str)

        if save_PNG:
            if not os.path.exists(dir_out):
                os.mkdir(dir_out)
            fig.savefig(
                fname=os.path.join(dir_out, f'{fname_out}.png'),
                bbox_inches='tight',
                format='png',
                dpi=300,)

        if save_SVG:
            if not os.path.exists(dir_out):
                os.mkdir(dir_out)
            fig.savefig(
                fname=os.path.join(dir_out, f'{fname_out}.svg'),
                bbox_inches='tight',
                format='svg',
                dpi=300,)

        plt.close()


class CheckDf():
    ''' Check assumptions about longform master DFs '''

    @staticmethod
    def check_master(df_master:pd.DataFrame, measure_params, timings:list[str]=commons_config.timings, conditions:list[str]=['C','T'],) -> None:
        ''' Check if df_master meets all assumptions '''

        df_duplicate_rows = CheckDf.check_duplicate_rows(df_master)
        CheckDf.check_conditions(df_master, conditions)
        CheckDf.check_time(df_master, measure_params)
        CheckDf.check_timings(df_master, timings)

        return  df_duplicate_rows

    @staticmethod
    def check_duplicate_rows(df_master:pd.DataFrame, cols=commons_config.cols_checkduplicates) -> None:
        ''' Check if there are duplicate rows '''
        df = df_master[cols]
        df_duplicates = df[df.duplicated(keep=False)]
        return df_duplicates

    @staticmethod
    def check_conditions(df_master:pd.DataFrame, conditions) -> None:
        ''' Check if there is a condition for every tp except baseline '''
        assert all([condition in conditions for condition in df_master.condition])

    @staticmethod
    def check_time(df_master:pd.DataFrame, measure_params) -> None:
        ''' Check the 'time' column per measure based on its has_time flag:
            - has_time=True: every row for that measure must have a numeric
              (non-NaN) time value.
            - has_time=False: every row for that measure must have time=NaN.
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
    def check_timings(df_master:pd.DataFrame, timings:list[str]=commons_config.timings) -> None:
        ''' Check if all timing is one of the expected values:
            -bsl: measured only at baseline, eg. demographic variables
            -change: measured both before and after treatment, eg. depressions scores
            -in-dose: measured during dosing session, thus, have a value for time, eg. vitals
            -post_dose: measured only post-dosing, eg. measures of trip quality like MEQ
            -post_trt: measured only post-treatment, eg. treatment satisfaction
        '''

        assert isinstance(df_master, pd.DataFrame)
        assert all([timing in timings for timing in df_master.timing])


class Helpers():
    ''' Various helper functions '''

    @staticmethod
    def sig_marking(value:float) -> str:
        ''' Converts p-values to standard significance marks '''

        if 0.05 > value >= 0.01:
            return '*'
        elif 0.01 > value >= 0.001:
            return '**'
        elif 0.001 > value:
            return '***'
        else:
            return ''

    @staticmethod
    def has_time(df:pd.DataFrame, measure:str) -> bool:
        ''' Detects whether the given measure has time, i.e. is it measured at
            multiple timepoints during the tp or not.

            Args:
                - df (pd.DataFrame): longform master df
                - measure (str): name of measure

            Returns:
                - has_time (bool): has time?
        '''

        # Check if all rows of time is NaN / non-NaN
        is_all_number = False
        is_all_nan = False

        if all([math.isnan(time) for time in df.loc[(df.measure==measure)].time]):
            is_all_nan = True
        if all([not math.isnan(time) for time in df.loc[(df.measure==measure)].time]):
            is_all_number = True

        # Decide if measure has time
        if (is_all_nan is True) and (is_all_number is True):
            raise UndecidedHasTime(measure)
        elif (is_all_nan is False) and (is_all_number is True):
            has_time = True
        elif (is_all_nan is True) and (is_all_number is False):
            has_time = False
        elif (is_all_nan is False) and (is_all_number is False):
            raise UndecidedHasTime(measure)
        else:
            assert False

        return has_time

    @staticmethod
    def convert_toHAMD(score, scale, delta=False):
        ''' Converts scores to HAMD17 equivalent.

        Args:
            score (float): original score
            scale (str): convert from what scale to HAMD17 equivalent
            delta (bool): if True, then delta scores are converted

        Returns:
            hamd_score (pd.Dataframe): HAMD17 equivalent score
            err (str): error message if any
        '''

        assert isinstance(delta, bool)
        assert isinstance(scale, str)

        if (score in [math.nan, None]) or math.isnan(score):
            return math.nan, None

        if scale=='svMADRS': # scored the same way
            scale='MADRS'
        if scale=='BDI': # default version of the scale
            scale='BDI1'

        # Convert score to float
        if delta is True:
            sign = -1 if score < 0 else 1
            score = float(abs(score))
        else:
            sign = 1
            score = float(score)

        # Get the right eq score dict
        if delta is False:
            if f'{scale.upper()}_to_HAMD17' in eqscores_hamd.eqscores.keys():
                xy_pairs=eqscores_hamd.eqscores[f'{scale.upper()}_to_HAMD17']
            else:
                return math.nan, f'HAMD17 eq scores are not defined for scale {scale.upper()}; converts to math.nan'
        else:
            if f'Δ{scale.upper()}_to_ΔHAMD17' in eqscores_hamd.eqscores.keys():
                xy_pairs=eqscores_hamd.eqscores[f'Δ{scale.upper()}_to_ΔHAMD17']
            else:
                return math.nan, f'HAMD17 eq Δ scores are not defined for scale {scale.upper()}; converts to math.nan'

        # Sort the eq score dictionary
        sorted_pairs = sorted(xy_pairs.items())
        x_vals = [pair[0] for pair in sorted_pairs]
        y_vals = [pair[1] for pair in sorted_pairs]

        if score < x_vals[0]:
            return math.nan, 'Some scores are below the defined minimum and cannot be converted to HAMD17; converts to math.nan'
        elif score > x_vals[-1]:
            return math.nan, 'Some scores are above the defined minimum and cannot be converted to HAMD17; converts to math.nan'

        ### Interpolate between the points
        for i in range(len(x_vals) - 1):
            if x_vals[i] <= score < x_vals[i+1]:
                hamd_score = round((y_vals[i] + (y_vals[i+1] - y_vals[i]) * (score - x_vals[i]) / (x_vals[i+1] - x_vals[i])), 2)
                break

        return (sign*hamd_score), None


class UndecidedHasTime(Exception):
    def __init__(self, measure):
        self.measure = measure
        super().__init__(self.measure)

class IncompleteScore(Exception):
    def __init__(self, msg, df_exception):
        self.msg = msg
        self.df_exception = df_exception
        super().__init__(self.msg, self.df_exception)
