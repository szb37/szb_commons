import src.config as config
import commons_codebase.src.config as commons_config
from statistics import mean, stdev
from scipy import stats
import matplotlib.pyplot as plt
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
    def get_df_measure(df_redcap:pd.DataFrame, measure_param:dict, cols_to_keep:list[str]=commons_config.cols_to_keep, **save)-> pd.DataFrame:
        ''' Returns all completed scores of a given measure in long-formatted df.
            Rows that have missing value in either pID/tp/score columns are removed.

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

    @staticmethod # DONE
    def format_bsl_vitals(df_redcap:pd.DataFrame, **save) -> pd.DataFrame:
        """ Deal with inconcistsent naming convention between baseline and post-baseline measures
            Need to call this before get_df_vitals().

            Args:
                - df_redcap (pd.DataFrame): raw REDCAP export df
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df_vitals (pd.DataFrame): long-form dataframe of vitals data
        """

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
        """ Special case of get_df_measure() to deal with the idiosyncrasies of vitals measures.
            Speifically, there is either 1 or 2 readings of vitals.
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
        """

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
    def add_sum_scores(df_redcap:pd.DataFrame, col_complete:str, col_items:list[str], col_score:str, **normalize) -> pd.DataFrame:
        ''' Calculates the sum of scores for columns in 'col_items' and adds the
            sum score to 'col_score' row of the input dataframe.
            Designed to work with wide format REDCap dfs.

            Args:
                - df_redcap (pd.DataFrame): raw export from REDCap
                - col_items (list of strs): list of columns that are summed
                - col_complete (str): name of column that defined whether row is complete
                - col_score (str): name of column where the sum scores are added
                - normalize (dict; optional): dictionary defining how to normalize the calculated sum scores

            Returns:
                - df_redcap (pd.DataFrame): REDCap df with col_score added
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(col_items, list)
        assert all([col_item in df_redcap.columns for col_item in col_items])
        assert isinstance(col_score, str)
        assert col_complete in df_redcap.columns

        # Check if summed columns do not have missing data
        ridx_missingitems = []
        for row in df_redcap.loc[(df_redcap[col_complete]==2)].itertuples():
            for col in col_items:
                if (eval(f'row.{col}') is None) or (math.isnan(eval(f'row.{col}'))):
                    ridx_missingitems.append(row.Index)

        if ridx_missingitems!=[]:
            ridx_missingitems = [ridx for ridx in set(ridx_missingitems)]
            print(f"\nMissing items from sum score calculation at row index (will skip rows from sum scores): {ridx_missingitems} \
                \n\tFirst summed column: {col_items[0]}")

        # Sum scores
        df_redcap.loc[(df_redcap[col_complete]==2), col_score] = df_redcap.loc[(df_redcap[col_complete]==2), col_items].sum(axis=1)
        df_redcap.loc[ridx_missingitems, col_score] = math.nan

        # Normalize sum scores if needed
        if normalize!={}:
            if normalize['normalize']=='by_nitems':
                norm_factor = len(col_items)
            elif normalize['normalize']=='by_maxscore':
                norm_factor = len(col_items)*normalize['max_item_score']
            else:
                assert False

            df_redcap.loc[(df_redcap[col_complete]==2), col_score] = round(df_redcap.loc[(df_redcap[col_complete]==2), col_score]/norm_factor, 3)

        return df_redcap

    @staticmethod
    def add_delta_scores(df_master:pd.DataFrame, delta_from_tp:str='bsl', delta_from_time:int=0) -> pd.DataFrame:
        ''' For every pID, tp, measure triplet add delta_score from timepoint defined by delta_from_tp if the row's measure has no time (i.e. all rows have time=nan)
            For every pID, tp, measure triplet add delta_score from the time defined by delta_from_time at the given timepoint if the row's measure has time (i.e. all rows have a non-nan time)

            WARNING: not optimized, may take a few mins with larger dfs

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
    def widen_master(df_master:pd.DataFrame, measures1:list[str], tp1:str, use_delta1:bool, measures2:list[str], tp2:str, use_delta2:bool) -> pd.DataFrame:
        ''' Convert long-form master df to wide-format df

            Args:
                - df_master(pd.DataFrame): master df of the trial
                - measures1(list[str]): list of measures, i.e. one set of column headers in the resulting wide-format df
                - tp1(str): use scores from what timepoint for measures in the measures1 list
                - use_delta1(bool): use delta_score/score in the measure's column for measures in the measures1 list
                - measures2(list[str]): list of measures, i.e. one set of column headers in the resulting wide-format df
                - tp2(str): use scores from what timepoint for measures in the measures2 list
                - use_delta2(bool): use delta_score/score in the measure's column for measures in the measures1 list

            Return:
                - df(pd.DataFrame): wide-format data frame
        '''

        assert isinstance(df_master, pd.DataFrame)
        for idx in [1,2]:
            assert isinstance(eval(f'measures{idx}'), list)
            assert sum([isinstance(measure, str) for measure in eval(f'measures{idx}')])
            assert isinstance(eval(f'tp{idx}'), str)

        df = df_master.loc[
            ((df_master.tp==tp1) & (df_master.measure.isin(measures1))) |
            ((df_master.tp==tp2) & (df_master.measure.isin(measures2)))]

        if use_delta1:
            df.loc[(df.tp==tp1) & (df.measure.isin(measures1)), 'score'] = df.loc[(df.tp==tp1) & (df.measure.isin(measures1)), 'delta_score']

        if use_delta2:
            df.loc[(df.tp==tp2) & (df.measure.isin(measures2)), 'score'] = df.loc[(df.tp==tp2) & (df.measure.isin(measures2)), 'delta_score']

        df = pd.pivot_table(df, index=['pID',], columns='measure', values='score', dropna=False)
        df.reset_index(inplace=True)

        return df


class Analysis():
    ''' Functions for data analysis '''

    @staticmethod
    def get_df_observed(df_master:pd.DataFrame, digits:int=3, **save) -> pd.DataFrame:
        """ Creates a dataframe with the observed mean and SD of all measures at every tp.
            Missing data are ignored from the mean/sd calculations.

            Args:
                - df_master (pd.DataFrame): long-form master dataframe containing all data
                - digits (int): round mean and SD to how many digits?
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df_observed: df of observed means and SDs at every tp
        """

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
    def get_df_tp_ndays(df_redcap:pd.DataFrame, **save) -> pd.DataFrame:
        """ Calculates the average number of days for each timepoint since baseline.
            This data is usefull when determining spacing between timepoints on various graphs

            Args:
                - df_redcap: raw REDCAP export df
                - save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                - df_tp_ndays: dataframe of timepoints and the avg days since baseline
        """

        ### Clean REDCap df
        df = df_redcap.rename(columns={'vrecord_date': 'date',})
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
    def get_corrmats(df:pd.DataFrame, measures1:list[str], measures2:list[str], methods:list[str]=commons_config.corr_methods, **save):
        """ Calculates and corr coeffs and associated p-values between all pairs of measures1 and measures2
            Correlations are calculated with 'pearson', 'spearman' and 'kendall' methods

            Args:
                df (pd.DataFrame): wide-format dataframe where all elements of measures1 and measures2 are columns
                measures1 (list of strs): variables for the x-axis of corr matrix
                measures2 (list of strs): variables for the y-axis of corr matrix
                methods (list of strs): what correlation method to use; elements must be pearson/spearman/kendall
                save (dict; optional): dictionary with keys dir_out fname_out that determine where output is saved

            Returns:
                df_coeffs (pd.DataFrame): dataframe of correlation coefficients
                df_pvalues (pd.DataFrame): dataframe of correlation p-values
        """

        assert isinstance(df, pd.DataFrame)
        for idx in [1,2]:
            assert isinstance(eval(f'measures{idx}'), list)
            assert sum([isinstance(var, str) for var in eval(f'measures{idx}')])
            assert sum([var in df.columns for var in eval(f'measures{idx}')])

        df_coeffs = pd.DataFrame(columns=measures1, index=measures2)
        df_pvalues = pd.DataFrame(columns=measures1, index=measures2)

        for method in methods:
            for var1, var2 in itertools.product(measures1, measures2):

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

            # Convert to numeric
            df_coeffs = df_coeffs.astype('float64')
            df_pvalues = df_pvalues.astype('float64')

            if save!={}:
                df_coeffs.to_csv(os.path.join(save['dir_out'], f'{save['fname_out']}_{method}_coeffs.csv'))
                df_pvalues.to_csv(os.path.join(save['dir_out'], f'{save['fname_out']}_{method}_pvalues.csv'))

                if save['draw']:
                    title = save['title'] if 'title' in save else None
                    Plots.draw_corrmat(
                        df_coeffs = df_coeffs,
                        df_pvalues = df_pvalues,
                        dir_out = save['dir_out'],
                        fname_out = f'{save['fname_out']}_{method}',
                        save = True,
                        title = title)

        return df_coeffs, df_pvalues


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
    def draw_corrmat(df_coeffs, df_pvalues, dir_out, fname_out, save=True, **kwargs):
        ''' Draws correlation matrix heatmap using outputs of get_corrmat()

            Args:
                df_coeffs (pd.DataFrame): dataframe of correlation coefficients
                df_pvalues (pd.DataFrame): dataframe of correlation p-values
                dir_out (str): where to save fig
                fname_out (str): name of file saves
                save (bool): should image be save
        '''

        assert isinstance(df_coeffs, pd.DataFrame)
        assert isinstance(df_pvalues, pd.DataFrame)
        assert isinstance(save, bool)
        assert isinstance(dir_out, str)
        assert isinstance(fname_out, str)

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

        plt.xticks(rotation=45)

        if 'title' in kwargs:
            ax.set_title(kwargs['title'], fontdict=config.title_fontdict)

        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'], fontdict=config.axislabel_fontdict)

        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'], fontdict=config.axislabel_fontdict)

        if save:
            Plots.save_fig(
                fig = fig,
                dir_out  = dir_out,
                fname_out = fname_out,
                save_PNG = config.save_PNG,
                save_SVG = config.save_SVG,)

    @staticmethod
    def draw_vitals(df_master, dir_out, prefix_out, save=True, measures=['dia', 'sys', 'hr'], **kwargs):
        """ Draw in-dosing-session trajectory of vitals
            Args:
                df_master (pd.DataFrame): long-form master df
                measures (list): list of vitals;
                dir_out (str): where to save results
                save (bool): save figure?
        """

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
                #style = 'tp',
                linewidth=2,
                markers = [
                    "o", "D"],
                palette = {
                    #'A0': '#56A0FB',
                    #'B0': '#F71480'},
                    'C': '#56A0FB',
                    'T': '#F71480'},
                errorbar = "ci",
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

            if save:
                Plots.save_fig(
                    fig = fig,
                    dir_out = dir_out,
                    fname_out = f'{prefix_out}_{measure}',
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
    def check_masterDf(df_master:pd.DataFrame, measure_types:list[str]=commons_config.measure_types) -> None:
        ''' Check if df_master meets all assumptions '''

        CheckDf.check_duplicate_rows(df_master)
        CheckDf.check_baseline_condition(df_master)
        CheckDf.check_indose_time(df_master)
        CheckDf.check_score_delta_score(df_master)
        CheckDf.check_measure_types(df_master, measure_types)

    @staticmethod
    def check_duplicate_rows(df_master:pd.DataFrame, cols=commons_config.cols_checkduplicates) -> None:
        ''' Check if there are duplicate rows '''

        df_master = df_master[cols]
        duplicate_rows = df_master[df_master.duplicated(keep=False)]
        if duplicate_rows.shape[0]!=0:
            print(f'There are {duplicate_rows.shape[0]} duplicate rows across {cols}.')
            print(duplicate_rows)

    @staticmethod
    def check_baseline_condition(df_master:pd.DataFrame) -> None:
        ''' Check if there is a condition for every tp except baseline '''

        assert isinstance(df_master, pd.DataFrame)
        assert all([condition in [None, ''] for condition in df_master.loc[(df_master.tp=='bsl')].condition])
        assert all([isinstance(condition, str) for condition in df_master.loc[(df_master.tp!='bsl')].condition])

    @staticmethod
    def check_indose_time(df_master:pd.DataFrame) -> None:
        ''' Check if there is time for all in_dose measures and that there is
            no time for not in_dose measures
        '''

        assert all(math.isnan(time) for time in df_master.loc[(df_master.measure_type!='in_dose')].time.tolist())
        assert all(isinstance(time, float) for time in df_master.loc[(df_master.measure_type=='in_dose')].time.tolist())

    @staticmethod
    def check_score_delta_score(df_master:pd.DataFrame) -> None:
        ''' Throws error if there is a delta_score for measures with meaure_type=post_dose and for all measures at baseline.
        '''
        assert isinstance(df_master, pd.DataFrame)
        assert all([((isinstance(score, float)) or (isinstance(score, int))) for score in df_master.score])
        assert all([((isinstance(delta_score, float)) or (isinstance(delta_score, int))) for delta_score in df_master.delta_score])

        # There should be no delta_score for post_dose measures and at baseline
        assert all([math.isnan(delta_score) for delta_score in df_master.loc[(df_master.measure_type=='post_dose')].delta_score])
        assert all([delta_score==0 for delta_score in df_master.loc[(df_master.tp=='bsl')].delta_score])

        missing_baselines = df_master.loc[(df_master.measure_type=='change') & pd.isna(df_master.delta_score)]
        if missing_baselines.shape[0]!=0:
            print("\nMissing delta_scores for following 'change' instruments (baseline missing?):")
            print(missing_baselines)

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
        is_all_not_nan = False
        is_all_nan = False

        if all([math.isnan(time) for time in df.loc[(df.measure==measure)].time]):
            is_all_nan = True
        if all([not math.isnan(time) for time in df.loc[(df.measure==measure)].time]):
            is_all_not_nan = True

        # Decide if measure has time
        if (is_all_nan is True) and (is_all_not_nan is True):
            raise UndecidedHasTime(measure)
        elif (is_all_nan is False) and (is_all_not_nan is True):
            has_time = True
        elif (is_all_nan is True) and (is_all_not_nan is False):
            has_time = False
        elif (is_all_nan is False) and (is_all_not_nan is False):
            raise UndecidedHasTime(measure)
        else:
            assert False

        return has_time


class UndecidedHasTime(Exception):
    def __init__(self, measure):
        self.measure = measure
        super().__init__(self.measure)

class MissingItemsFromSumScore(Exception):
    def __init__(self, msg, df_exception):
        self.msg = msg
        self.df_exception = df_exception
        super().__init__(self.msg, self.df_exception)
