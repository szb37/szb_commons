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
    def clean_df_redcap(df_redcap:pd.DataFrame, rename_tps:dict={}, n_prefix_chars:int=0, rm_tps:list=[], convert_pids_int:bool=True) -> pd.DataFrame:
        """
        Clean df_redcap by optionally renaming timepoints, removing specified timepoints, and converting pID to integer.

        Args:
            df_redcap (pd.DataFrame): Raw REDCap export dataframe.
            rename_tps (dict, optional): Dictionary mapping old to new timepoint names (e.g., {'tp_old': 'tp_new'}). Default is {} (no renaming).
            n_prefix_chars (int, optional): Number of characters to remove from the start of each pID value before converting to integer. Default is 0.
            rm_tps (list, optional): List of timepoints to remove from the dataframe. Rows with tp in this list will be dropped. Default is [].
            convert_pids_int (bool, optional): Whether to convert pID to int and remove rows with non-convertible pIDs. Default is True.

        Returns:
            pd.DataFrame: Cleaned REDCap dataframe.
        """

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(rename_tps, dict)
        assert isinstance(n_prefix_chars, int)
        assert isinstance(rm_tps, list)
        assert isinstance( convert_pids_int, bool)

        ### Rename columns & tps; remove spurious timepoints
        df_redcap = df_redcap.rename(columns={
            'participant_id': 'pID',
            'record_id': 'pID',
            'redcap_event_name': 'tp',})

        if rename_tps!={}:
            df_redcap['tp'] = df_redcap['tp'].replace(rename_tps)

        if rm_tps != []:
            df_redcap = df_redcap.loc[~df_redcap.tp.isin(rm_tps)]

        # Remove pID prefixes prior to INT conversion
        if n_prefix_chars > 0:
            df_redcap['pID'] = df_redcap['pID'].astype(str).str[n_prefix_chars:]

        # Convert pID to numeric, print & delete non-convertible pIDs 
        if convert_pids_int:
            pID_numeric = pd.to_numeric(df_redcap['pID'], errors='coerce')
            bad_pids = df_redcap.loc[pID_numeric.isna(), 'pID'].unique()
            if len(bad_pids) > 0:
                print(f'Unique pIDs that could not be converted to int (deleting rows):')
                for bad_pid in bad_pids:
                    print(f'\t{bad_pid}')

            df_redcap = df_redcap.loc[~pID_numeric.isna()].copy()
            df_redcap['pID'] = pID_numeric[~pID_numeric.isna()].astype(int)

        return df_redcap

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
            ['instrument', 'measure', 'measure_type', 'col_complete', 'col_score',]])
        assert isinstance(cols_to_keep, list)

        # define variables
        df = df_redcap.copy()
        col_score = measure_param['col_score']
        col_complete = measure_param['col_complete']

        # reduce dataframe
        if col_complete is not None:
            df = df.loc[(df[col_complete]==2)]
        df = df[cols_to_keep + [col_score]]

        # rename / add bookkeeping columns
        df.rename(columns={col_score: 'score'}, inplace=True)
        df['instrument'] = measure_param['instrument']
        df['measure'] = measure_param['measure']
        df['measure_type'] = measure_param['measure_type']

        # deal with time
        if 'time' in measure_param.keys():
            df['time'] = measure_param['time']
        else:
            df['time'] = math.nan

        ### Cleanup, save, return output
        df = df.dropna(subset=(['pID', 'tp', 'score']))
        df['score'] = df['score'].astype('float64')
        df['time'] = df['time'].astype('float64')
        df = df[cols_to_keep+['measure_type','instrument','measure','time','score']]
        df.reset_index(inplace=True, drop=True)

        if save!={}:
            df.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df

    @staticmethod
    def get_df_tp_ndays(df_redcap:pd.DataFrame, col_date:str, **save) -> pd.DataFrame:
        ''' Get df of the average number of days since baseline for each timepoint.

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

            ### Create row with avg number of days between baseline and tp
            df_tp_ndays_rows.append([
                tp,
                round(np.array(ndays).mean()),
                len(ndays)])

        ### Convert list of rows to df
        df_tp_ndays = pd.DataFrame(columns=['tp', 'ndays', 'n'], data=df_tp_ndays_rows)
        df_tp_ndays.sort_values(by='ndays', inplace=True)

        ### Save if needed, return output
        if save!={}:
            df_tp_ndays.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_tp_ndays

    @staticmethod
    def get_df_dosedates(df_redcap, col_date, dose_tps=['A0', 'B0', 'C0', 'D0',], **save):
        ''' Get df of when each dose was administered
            Args:
                - df_redcap (pd.DataFram): REDCAP export
                - col_date (str): column of dates in REDCap df
                - dose_tps (list): list of timepoints to include in result

            Returns:
                - df_dosedates (pd.DataFrame): df of date-dose pairs
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(col_date, str)

        df_dosedates = df_redcap.loc[(df_redcap.tp.isin(dose_tps))]
        df_dosedates = df_dosedates.rename(columns={col_date: 'my_date',})
        df_dosedates = df_dosedates[['pID', 'tp', 'my_date']]
        df_dosedates = df_dosedates.dropna()
        df_dosedates = df_dosedates.drop_duplicates()
        df_dosedates['my_date'] = pd.to_datetime(df_dosedates['my_date'])
        df_dosedates = df_dosedates.rename(columns={'my_date': 'date',})
        df_dosedates = df_dosedates.reset_index(drop=True)

        if save!={}:
            df_dosedates.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_dosedates

    @staticmethod
    def format_bsl_vitals(df_redcap:pd.DataFrame, **save) -> pd.DataFrame:
        ''' Deal with inconcistsent naming convention between baseline and post-baseline vitals measures.
            Need to call this before get_df_vitals().

            Args:
                - df_redcap (pd.DataFrame): raw REDCAP export df
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df_vitals (pd.DataFrame): long-form dataframe of vitals data
        '''

        assert isinstance(df_redcap, pd.DataFrame)

        # Change name of baseline columns to fit naming convention of other columns
        df_redcap = df_redcap.rename(
            columns={
                'vs_bl_dia': 'vs_dose0_dia1',
                'vs_bl_sys': 'vs_dose0_sys1',
                'vs_bl_hr': 'vs_dose0_hr1',}, inplace=False)

        df_redcap.loc[:, 'vs_dose0_dia1'] = pd.to_numeric(df_redcap['vs_dose0_dia1'])
        df_redcap.loc[:, 'vs_dose0_sys1'] = pd.to_numeric(df_redcap['vs_dose0_sys1'])
        df_redcap.loc[:, 'vs_dose0_hr1'] = pd.to_numeric(df_redcap['vs_dose0_hr1'])

        # create second reading for bsl measures for consistency
        df_redcap.insert(loc=df_redcap.columns.get_loc('vs_dose0_dia1')+1, column='vs_dose0_dia2', value=math.nan)
        df_redcap.insert(loc=df_redcap.columns.get_loc('vs_dose0_sys1')+1, column='vs_dose0_sys2', value=math.nan)
        df_redcap.insert(loc=df_redcap.columns.get_loc('vs_dose0_hr1')+1, column='vs_dose0_hr2', value=math.nan)

        ### Save, return output
        if save!={}:
            df_redcap.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_redcap

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
            ['instrument', 'measure', 'measure_type', 'col_complete', 'col_score',]])

        df = df_redcap.loc[(df_redcap[measure_param['col_complete']]==2)]
        rows_vitals = []

        for row in df.itertuples():
            pID = row.pID
            tp = row.tp

            score1 = eval(f'row.{measure_param['col_score']}1')
            score2 = eval(f'row.{measure_param['col_score']}2')

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
                'in_dose',
                measure_param['instrument'],
                measure_param['measure'],
                measure_param['time'],
                score,])

        df_vitals = pd.DataFrame(columns=[
            'pID',
            'tp',
            'measure_type',
            'instrument',
            'measure',
            'time',
            'score',], data=rows_vitals)

        ### Cleanup, save, return output
        df_vitals.dropna(subset=['pID', 'tp', 'score'], inplace=True)
        df_vitals['score'] = df_vitals['score'].astype('float64')
        df_vitals['time'] = df_vitals['time'].astype('float64')
        df_vitals = df_vitals[['pID','tp','measure_type','instrument','measure','time','score']]
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

    @staticmethod
    def calc_scores(df_redcap:pd.DataFrame, measure_param:dict) -> pd.DataFrame:
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
        col_items = measure_param['col_items']
        col_score = measure_param['col_score']

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(col_items, list)
        assert all([col_item in df_redcap.columns for col_item in col_items])
        assert isinstance(col_score, str)
        assert col_complete in df_redcap.columns

        ### Reverse items if there is any
        if 'reverse_items' in measure_param:
            assert all([limit in measure_param for limit in ['min_score', 'max_score']])
            original_scores = df_redcap.loc[(df_redcap[col_complete]==2), measure_param['reverse_items']]
            reverse_scores = \
                (measure_param['max_score'] + measure_param['min_score']) - \
                df_redcap.loc[(df_redcap[col_complete]==2), measure_param['reverse_items']]
            df_redcap.loc[(df_redcap[col_complete]==2), measure_param['reverse_items']] = reverse_scores

        ### Calculate sum scores
        df_redcap.loc[(df_redcap[col_complete]==2), col_score] = df_redcap.loc[(df_redcap[col_complete]==2), col_items].sum(axis=1)

        ### Reverse back reverse-items back to original as the column may get processed again in different contexts
        if 'reverse_items' in measure_param:
            df_redcap.loc[(df_redcap[col_complete]==2), measure_param['reverse_items']] = original_scores

        ### Normalize sum scores
        if 'norm_factor' in measure_param:
            df_redcap.loc[(df_redcap[col_complete]==2), col_score] = \
            df_redcap.loc[(df_redcap[col_complete]==2), col_score]/measure_param['norm_factor']

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
        ''' For every pID, tp, measure triplet add delta_score from timepoint defined by delta_from_tp if the row's measure has no time (i.e. all rows have time=nan)
            For every pID, tp, measure triplet add delta_score from the time defined by delta_from_time at the given timepoint if the row's measure has time (i.e. all rows have a non-nan time)

            TODO: optimize, right now its brute force. May take some time for larger dfs

            Args:
                - df (pd.DataFrame): longform df of trial data
                - delta_from_tp (str): what value in "tp" designates baseline
                - delta_from_tp (str): what value in "time" designates start

            Returns:
                - df (pd.DataFrame): longform df of trial data with delta_score added
        '''

        assert isinstance(df_master, pd.DataFrame)
        assert isinstance(delta_from_tp, str)
        assert isinstance(delta_from_time, int)

        df_master['delta_score'] = math.nan
        cidx_score = df_master.columns.get_loc('score')
        cidx_dltscore = df_master.columns.get_loc('delta_score')
        undecided_has_time=[]

        for row in df_master.itertuples():

            try:
                has_time = Helpers.has_time(df_master, row.measure)
            except UndecidedHasTime as e:
                undecided_has_time.append(e.measure)
                continue

            # Find baseline value
            if has_time:
                ridx_bsl = df_master.loc[
                    (df_master.pID == row.pID) &
                    (df_master.measure == row.measure) &
                    (df_master.tp == row.tp) &
                    (df_master.time == delta_from_time)].index
            else:
                ridx_bsl = df_master.loc[
                    (df_master.pID == row.pID) &
                    (df_master.measure == row.measure) &
                    (df_master.tp == delta_from_tp)].index

            assert ((len(ridx_bsl)==1) or (len(ridx_bsl)==0))

            if len(ridx_bsl)==0:
                continue
            else:
                ridx_bsl = ridx_bsl[0]

            # Add delta score
            bsl_score = df_master.iloc[ridx_bsl, cidx_score]
            tp_score = row.score
            df_master.iloc[row.Index, cidx_dltscore] = tp_score-bsl_score

        # Warn user
        if len(undecided_has_time)!=0:
            undecided_has_time = set(undecided_has_time)
            print(f"Can not decide whether measure has time: {[measure for measure in undecided_has_time]}")

        return df_master

    @staticmethod
    def get_df_aes(df_redcap, df_redcap_datalabels, **save):
        
        # Switching out meddra codes for titles
        cidx_meddra = df_redcap.columns.get_loc('ae_1')
        df_redcap.iloc[:, cidx_meddra] = df_redcap_datalabels.iloc[:, cidx_meddra]
        
        df_ae = df_redcap[df_redcap.adverse_event_log_complete==2]
        #df_ae = df_ae[~df_ae['pID'].str.contains('test', case=False)]

        ### Rename columns
        df_ae = df_ae[["pID", "tp"] + [col for col in df_ae.columns if "ae_" in col]]
        df_ae = df_ae.rename(columns={
            'ae_1':  'meddra',
            'ae_8':  'description',
            'ae_2':  'category',
            'ae_10': 'start_date',
            'ae_13': 'last_dose_date',
            'ae_13a___1': 'is_predrug',
            'ae_16': 'was_intervention',
            'ae_9':  'intervention_desc',
            'ae_19': 'outcome', 
            'ae_20': 'severity',
            'ae_21': 'is_serious',
            'ae_22': 'outcome_of_serious',
            'ae_26': 'related_drug',
            'ae_27': 'related_procedures',
            'ae_28': 'action',
            'ae_30': 'date_attestation',})

        ### Set end_date to the most recent date among follow-ups
        df_ae['end_date'] = None
        fu_cols = [f"ae_fu{i}_8" for i in range(1, 11) if f"ae_fu{i}_8" in df_ae.columns] + ['ae_12']
        df_ae['end_date'] = df_ae[fu_cols].apply(lambda row: pd.to_datetime(row, errors='coerce').max(), axis=1)

        ### Compute is_expected_combined using ae_24 and ae_25
        conds = [
            (df_ae['ae_24'] == 1) & (df_ae['ae_25'] == 1),
            (df_ae['ae_24'] == 1) & (df_ae['ae_25'] == 2),
            (df_ae['ae_24'] == 1) & (df_ae['ae_25'].isna()),
            (df_ae['ae_24'] == 2)]
        choices = [
            'Expected result of study drug(s)',
            'Expected result of study procedure(s)',
            'Expected, but neither "expected of drug" nor "expected of procedure" was selected',
            'Not Expected']
        df_ae['is_expected'] = np.select(conds, choices, default='Not Expected')

        ### Compute serious outcomes
        df_ae['outcome_serious'] = df_ae.apply(DataWrangl.get_outcome_serious, axis=1)

        ### Recode responses
        df_ae["severity"] = df_ae["severity"].replace({
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Life-threatening"})
        df_ae["category"] = df_ae["category"].replace({
            1: "Cardiovascular",
            2: "Respiratory",
            3: "Gastrointestinal",
            4: "Genitourinary",
            5: "Musculoskeletal",
            6: "Dermatologic",
            7: "Neurologic",
            8: "Hematologic"})
        df_ae["related_drug"] = df_ae["related_drug"].replace({
            0: "Not Related",
            1: "Possible",
            2: "Probable",
            3: "Definite"})
        df_ae["related_procedures"] = df_ae["related_procedures"].replace({
            0: "Not Related",
            1: "Possible",
            2: "Probable",
            3: "Definite"})
        df_ae["action"] = df_ae["action"].replace({
            1: "PI has decided to withdraw the participant from the study.",
            2: "PI has decided to withhold further drug administration.",
            3: "Participant has decided to withdraw from the study.",
            4: "No action on enrollment by study team or participant.",}) 
        df_ae["outcome"] = df_ae["outcome"].replace({
            1: "Life-threatening/Fatal",
            2: "Chronic/not expected to recover",
            3: "Expected to recover prior to end of participation",
            4: "Expected to recover after end of participation",
            5: "Recovered at time of initial report",}) 

        ### Organize columns
        df_ae = df_ae[[
            'pID', 'meddra', 'category', 'description', 
            'severity', 'is_serious', 
            'was_intervention', 'intervention_desc',
            'is_expected', 'related_drug', 'related_procedures',
            'start_date', 'end_date', 'last_dose_date', 'is_predrug', 'action','outcome', 'outcome_serious',
            ]]

        if save!={}: 
            df_ae.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)

        return df_ae

    @staticmethod
    def get_outcome_serious(row):
        serious_map = {
            'ae_22___1': 'Death',
            'ae_22___2': 'Life-threatening',
            'ae_22___3': 'Requires or prolongs hospitalization (does not include ED visits w/o admission)',
            'ae_22___4': 'Disability or permanent damage',
            'ae_22___5': 'Congenital abnormality/birth defect or cancer',
            'ae_22___6': 'Required intervention (medical or surgical) to prevent permanent impairment or damage or prevent outcome listed above'}
        outcomes = [serious_map[col] for col in serious_map if row.get(col, 0) == 1]
        return '; '.join(outcomes) if outcomes else ''


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
                round(mean(scores), digits),
                round(stdev(scores), digits),])

        ### Create DF from list of rows
        df_observed = pd.DataFrame(
            columns=['measure','tp','mean','sd'],
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

        df_res = pd.DataFrame(columns=['pID', 'tp', 'measure',])

        for measure_param in measure_params:

            if 'tps' not in measure_param.keys():
                print(f'tps are not defined for {measure_param['measure']}; cannot find missing measurs.')
                continue

            for pID, tp in itertools.product(df_master.pID.unique(), measure_param['tps']):
                score = df_master.loc[(df_master.pID==pID) & (df_master.tp==tp) & (df_master.measure==measure_param['measure'])].score
                nrows = df_master.loc[(df_master.pID==pID) & (df_master.tp==tp) & (df_master.measure==measure_param['measure'])].shape[0]

                if (nrows==1) and (not math.isnan(score)):
                    continue

                df_res.loc[df_res.shape[0]] = {
                	'pID': pID,
                	'tp': tp,
                	'measure': measure_param['measure'],}

        if save!={}:
            df_res.to_csv(os.path.join(save['dir_out'], save['fname_out']), index=False)
            if df_res.shape[0] > 0:
                print(f'Some measures are missing, see {save['fname_out']} for details.')

        return df_res

    @staticmethod
    def insert_HAMDequal_score(df, cols, delta=False):
        ''' Inserts new columns with the HAMD17 equivalent of cols.
            The equivalent scores will be in the new column '{col}_HAMD'

        Args:
            df (pd.Dataframe):
            cols (list of str): columns that are converted to HAMD17 equivalent
            delta (bool): if True, then delta scores are converted

        Returns:
            df (pd.Dataframe): the original dataframe with the cols_HAMD column(s) added
        '''

        assert isinstance(df, pd.DataFrame)
        assert isinstance(cols, list)
        assert isinstance(delta, bool)

        cidx_scale = df.columns.get_loc('scale')
        errors=[]

        for col in cols:

            df.insert(len(df.columns), f'{col}_HAMD', math.nan)
            cidx_original_col = df.columns.get_loc(col)
            cidx_hamd_eq = df.columns.get_loc(f'{col}_HAMD')

            for row in df.itertuples():

                if df.iloc[row.Index, cidx_original_col] in [None, math.nan]:
                    continue

                if row.scale in ['HAMD', 'HAMD17', 'GRID-HAMD',]:
                    # No need to convert, but double check if GRID-HAMD and HAMD refer to the 17 item version!
                    df.iloc[row.Index, cidx_hamd_eq] = df.iloc[row.Index, cidx_original_col]
                else:
                    # attempt to convert
                    hamd_eq, err = Helpers.convert_toHAMD(
                        df.iloc[row.Index, cidx_original_col],
                        df.iloc[row.Index, cidx_scale],
                        delta=delta)

                    df.iloc[row.Index, cidx_hamd_eq] = hamd_eq
                    if err is not None:
                        errors.append(err)

            # Round results
            df[f'{col}_HAMD'] = round(df[f'{col}_HAMD'].copy(), 3)

        # Display unique errors
        if errors!=[]:
            for err in set(errors):
                print(err)

        return df


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
    def check_master(df_master:pd.DataFrame, trial:str, folder_exports:str, measure_types:list[str]=commons_config.measure_types) -> None:
        ''' Check if df_master meets all assumptions '''

        CheckDf.check_duplicates(df_master, trial, folder_exports)
        CheckDf.check_conditions(df_master)
        CheckDf.check_indose_time(df_master)
        CheckDf.check_measure_types(df_master, measure_types)

    @staticmethod
    def check_duplicates(df_master:pd.DataFrame, trial:str, folder_exports:str, cols=commons_config.cols_checkduplicates) -> None:
        ''' Check if there are duplicate rows '''

        df_master = df_master[cols]
        duplicates = df_master[df_master.duplicated(keep=False)]
        if duplicates.shape[0]!=0:
            print(f'Duplicate rows, see exports/{trial}_duplicates.csv.')
            duplicates.to_csv(os.path.join(folder_exports, f'{trial}_duplicates.csv'), index=False)

    @staticmethod
    def check_conditions(df_master:pd.DataFrame, tps_wo_condition=['bsl', 'ltfu']) -> None:
        ''' Check if there is a condition for every tp except baseline '''

        assert all([condition in [None, ''] for condition in df_master.loc[(df_master.tp.isin(tps_wo_condition))].condition])
        assert all([isinstance(condition, str) for condition in df_master.loc[(~df_master.tp.isin(tps_wo_condition))].condition])

    @staticmethod
    def check_indose_time(df_master:pd.DataFrame) -> None:
        ''' Check if there is time for all in_dose measures and that there is
            no time for not in_dose measures
        '''

        assert all(math.isnan(time) for time in df_master.loc[(df_master.measure_type!='in_dose')].time.tolist())
        assert all(isinstance(time, float) for time in df_master.loc[(df_master.measure_type=='in_dose')].time.tolist())

    @staticmethod
    def check_measure_types(df_master:pd.DataFrame, measure_types:list[str]=commons_config.measure_types) -> None:
        ''' Check if all measure_type is one of the expected values:
            -bsl: measured only at baseline, eg. demographic variables
            -change: measured both before and after treatment, eg. depressions scores
            -in_dose: measured during dosing session, thus, have a value for time, eg. vitals
            -post_dose: measured only post-dosing, eg. measures of trip quality like MEQ
            -post_trt: measured only post-treatment, eg. treatment satisfaction
        '''

        assert isinstance(df_master, pd.DataFrame)
        assert all([measure_type in measure_types for measure_type in df_master.measure_type])


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
