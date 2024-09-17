import matplotlib.pyplot as plt
from statistics import mean, stdev
import src.config as commons_config
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import warnings
import math
import os


warnings.filterwarnings("ignore", category=DeprecationWarning)


class DataWrangl():
    ''' Functions for common data wrangling tasks '''

    @staticmethod
    def get_longdf_of_measure(df_redcap, measure, cols_to_keep=['pID', 'tp']):
        ''' Extracts the scores associated with 'measure' from the raw REDCap df,
            where 'measure' is a dictionary defining what columns are taken from
            REDCap df and how the output is formatted.

            Args:
                - df_redcap (pd.DataFrame): raw export from REDCap
                - cols_to_keep (list of strs): what columns should be kept in longform DF
                - measure (dict): dictionary that defines what columns are taken from df_redcap
                    and how the output is formatted. It is assumed that the dictionary
                    has the following keys (can have other keys as well):
                        - 'instrument': value of the "instrument" column in the returned df;
                        - 'measure': value of the "measure" column in the returned df; "measure" is typcially name of the scale or a subscale
                        - 'col_complete': column name in df_redcap, which tracks if row was completed; if set to None, completion is not checked, if a str is provided only rows are kept where its value is 2
                        - 'col_score': column name in df_redcap, which stores score
                        - 'type': value of the "type" column in the returned df; usefull to distinguish structure of measures

            Returns:
                - df_measure (pd.DataFrame): longform df with all scores of the defined measure
        '''

        assert isinstance(df_redcap, pd.DataFrame)
        assert isinstance(measure, dict)
        assert all([field in measure.keys() for field in
            ['instrument', 'measure', 'measure_type', 'col_complete', 'col_score']])
        assert isinstance(cols_to_keep, list)

        # define variables
        df_measure = df_redcap.copy()
        col_score = measure['col_score']
        col_complete = measure['col_complete']
        cols_to_keep = cols_to_keep + [col_score]

        # reduce dataframe
        if col_complete is not None:
            df_measure = df_measure.loc[(df_measure[col_complete]==2)]
        df_measure = df_measure[cols_to_keep]
        df_measure = df_measure.dropna(subset=[col_score])

        # rename / add bookkeeping columns
        df_measure.rename(columns={col_score: 'score'}, inplace=True)
        df_measure['score'] = df_measure['score'].astype('float64')
        df_measure['instrument'] = measure['instrument']
        df_measure['measure'] = measure['measure']
        df_measure['measure_type'] = measure['measure_type']

        return df_measure

    @staticmethod
    def add_sum_scores(df_redcap, col_complete, col_items, col_score, **kwargs):
        ''' Calculates the sum of scores for columns in 'col_items' and adds the
            sum score to 'col_score' row of the input dataframe.
            Designed to work with wide format REDCap dfs.

            Args:
                - df_redcap (pd.DataFrame): raw export from REDCap
                - col_items (list of strs): list of columns that are summed
                - col_complete (str): name of column that defined whether row is complete
                - col_score (str): name of column where the sum scores are added
                - norm (bool): should the value in col_score be normlaized, i.e. divided by the number of items in 'col_items'

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
        if 'normalize' in kwargs:

            if kwargs['normalize']=='by_item_number':
                norm_factor = len(col_items)
            elif kwargs['normalize']=='by_max_value':
                norm_factor = len(col_items)*kwargs['max_value']

            df_redcap.loc[(df_redcap[col_complete]==2), col_score] = df_redcap.loc[(df_redcap[col_complete]==2), col_score] / norm_factor

        return df_redcap

    @staticmethod
    def add_delta_scores(df_master, delta_from_tp='bsl', delta_from_time=0):
        ''' For every pID, tp, measure triplet add delta_score from the delta_from tp if time column of the measure is NaN
            For every pID, tp, measure triplet add delta_score from the delta_from tp if time column of the measure is not-NaN
            NOT OPTIMIZED, will take a bit for larger dfs

            Args:
                - df (pd.DataFrame): longform df of trial data
                - delta_from_tp (str): what value in "tp" designates baseline
                - delta_from_tp (str): what value in "time" designates start

            Returns:
                - df (pd.DataFrame): longform df of trial data with delta_score added
        '''

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


class Analysis():

    @staticmethod
    def get_df_observed(df_master, fname_out='bap1_observed_scores.csv', save=False, digits=1, **kwargs):
        """ Creates a dataframe with the observed mean±SD of all measures at every tp
            Args:
                - df_master (pd.DataFrame): long-form master dataframe containing all data
                - fname_out(str): name of output CSV saved at folders.exports
                - save (bool): save results?
                - digits (int): round mean and SD to how many digits?

            Returns:
                - df_observed(pd.DataFrame): dataframe of observed mean±SD of all measures at every tp
        """

        # get what measures and tps will be in df, initate df_observed
        if 'measures' in kwargs:
            measures = kwargs['measures']
        else:
            measures = df_master.measure.unique().tolist()

        if 'tps' in kwargs:
            tps = kwargs['tps']
        else:
            tps = df_master.tp.unique().tolist()

        df_observed = pd.DataFrame(columns=tps, index=measures)

        # loop through measures and tps, calculate mean±SD for each cell
        for measure, tp in itertools.product(measures, tps) :

            scores = df_master.loc[(df_master.measure==measure) & (df_master.tp==tp)].score.tolist()
            if len(scores)<3: # skip if not enough data to calc SD
                continue

            ridx = df_observed.index.get_loc(measure)
            cidx = df_observed.columns.get_loc(tp)

            # calculate mean±SD for appropiate number of digits
            if 'digits_measure' in kwargs:
                if measure in kwargs['digits_measure'].keys():
                    digits_tmp = kwargs['digits_measure'][measure] # set temporary number of digits
                    value_cell = f'{round(mean(scores), digits_tmp)}±{round(stdev(scores), digits_tmp)}'
                else:
                    value_cell = f'{round(mean(scores), digits)}±{round(stdev(scores), digits)}'
            else:
                value_cell = f'{round(mean(scores), digits)}±{round(stdev(scores), digits)}'

            df_observed.iloc[ridx, cidx] = value_cell

        # sort columns and rows
        df_observed = df_observed.sort_index()
        tps.sort(key=lambda x: (x != 'bsl', x))
        df_observed = df_observed[tps]

        # save & return
        if save:
            df_observed.to_csv(os.path.join(kwargs['dir_out'], fname_out), index=True, encoding='latin1')

        return df_observed

    @staticmethod
    def get_corrmats(df, vars1, vars2, dir_out, prefix_out, do_draw=True, save=False, **kwargs):
        """ Calculates and corr coeffs and associated p-values between all pairs of vars1 and vars2
            Correlations are calculated with 'pearson', 'spearman' and 'kendall' methods

            Args:
                df (pd.DataFrame): wide-format dataframe where all elements of vars1 and vars2 are columns
                vars1 (list of strs): predictor variables; x-axis of corr matrix
                vars2 (list of strs): outcome variables; y-axis of corr matrix
                save(boolean): save results?
                dir_out(str): string to folder where results saved
                prefix_out(str): filename prefix of the outputs

            Returns:
                df_coeffs (pd.DataFrame): dataframe of correlation coefficients
                df_pvalues (pd.DataFrame): dataframe of correlation p-values
        """

        assert isinstance(df, pd.DataFrame)
        for idx in [1,2]:
            assert isinstance(eval(f'vars{idx}'), list)
            assert sum([isinstance(var, str) for var in eval(f'vars{idx}')])
            assert sum([var in df.columns for var in eval(f'vars{idx}')])

        df_coeffs = pd.DataFrame(columns=vars1, index=vars2)
        df_pvalues = pd.DataFrame(columns=vars1, index=vars2)

        for method in ['pearson', 'spearman', 'kendall']:
            for var1, var2 in itertools.product(vars1, vars2):

                df_tmp = df[[var1, var2]]
                df_tmp = df_tmp.dropna()

                if method == 'pearson':
                     result_corr = stats.pearsonr(df_tmp[var1], df_tmp[var2])
                elif method == 'spearman':
                     result_corr = stats.spearmanr(df_tmp[var1], df_tmp[var2])
                elif method == 'kendall':
                     result_corr = stats.kendalltau(df_tmp[var1], df_tmp[var2])

                df_coeffs.at[var2, var1] = round(result_corr.statistic, 3)
                df_pvalues.at[var2, var1] = round(result_corr.pvalue, 3)

            if save:
                df_coeffs.to_csv(os.path.join(dir_out, f'{prefix_out}_{method}_coeffs.csv'), index=False)
                df_pvalues.to_csv(os.path.join(dir_out, f'{prefix_out}_{method}_pvalues.csv'), index=False)

            if do_draw:
                Plots.draw_corrmat(
                    df_coeffs = df_coeffs,
                    df_pvalues = df_pvalues,
                    dir_out = dir_out,
                    fname_out = f'{prefix_out}_{method}_plot',
                    save = save,
                    title = kwargs['title']+f' {method.upper()}')


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
            ax.set_title(kwargs['title'], fontdict=commons_config.title_fontdict)

        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'], fontdict=commons_config.axislabel_fontdict)

        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'], fontdict=commons_config.axislabel_fontdict)

        if save:
            Plots.save_fig(
                fig = fig,
                dir_out  = dir_out,
                fname_out = fname_out,
                save_PNG = commons_config.save_PNG,
                save_SVG = commons_config.save_SVG,)

    @staticmethod
    def save_fig(fig, dir_out, fname_out, save_PNG, save_SVG,):
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
            fig.savefig(
                fname=os.path.join(dir_out, f'{fname_out}.png'),
                bbox_inches='tight',
                format='png',
                dpi=300,)

        if save_SVG:
            fig.savefig(
                fname=os.path.join(dir_out, f'{fname_out}.svg'),
                bbox_inches='tight',
                format='svg',
                dpi=300,)

        plt.close()


class CheckDf():
    ''' Check assumptions about longform master DFs '''

    @staticmethod
    def check_masterDf(df_master, measure_types=['bsl', 'change', 'in_dose', 'post_dose']):
        ''' Check if longform master df_master for all assumptions '''

        assert isinstance(df_master, pd.DataFrame)

        CheckDf.check_duplicate_rows(df_master)
        CheckDf.check_baseline_condition(df_master)
        CheckDf.check_indose_time(df_master)
        CheckDf.check_score_delta_score(df_master)
        CheckDf.check_measure_types(df_master, measure_types)

    @staticmethod
    def check_duplicate_rows(df_master, cols=['pID', 'tp', 'measure', 'time']):
        ''' Check if there are duplicate rows '''

        assert isinstance(df_master, pd.DataFrame)

        df_master = df_master[cols]
        duplicate_rows = df_master[df_master.duplicated(keep=False)]
        if duplicate_rows.shape[0]!=0:
            print(f'There are {duplicate_rows.shape[0]} duplicate rows across {cols}.')
            print(duplicate_rows)

    @staticmethod
    def check_baseline_condition(df_master):
        ''' Check if there is a condition for every tp except baseline '''

        assert isinstance(df_master, pd.DataFrame)

        assert all([condition is None for condition in df_master.loc[(df_master.tp=='bsl')].condition])
        assert all([isinstance(condition, str) for condition in df_master.loc[(df_master.tp!='bsl')].condition])

    @staticmethod
    def check_indose_time(df_master):
        ''' Check if there is time for all in_dose measures and that there is
            no time for not in_dose measures
        '''

        assert isinstance(df_master, pd.DataFrame)

        assert all(math.isnan(time) for time in df_master.loc[(df_master.measure_type!='in_dose')].time.tolist())
        assert all(isinstance(time, float) for time in df_master.loc[(df_master.measure_type=='in_dose')].time.tolist())

    @staticmethod
    def check_score_delta_score(df_master):
        ''' Check if there is time for all in_dose measures and that there is
            no time for not in_dose measures
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
    def check_measure_types(df_master, measure_types):
        ''' Check if all measure_type is one of the defined values
        '''

        assert isinstance(df_master, pd.DataFrame)
        assert all([measure_type in measure_types for measure_type in df_master.measure_type])


class Helpers():
    ''' Various helper functions '''

    @staticmethod
    def sig_marking(value):
        ''' Converts p-values to standard significance marks '''

        assert isinstance(value, float)

        if 0.05 > value >= 0.01:
            return '*'
        elif 0.01 > value >= 0.001:
            return '**'
        elif 0.001 > value:
            return '***'
        else:
            return ''

    @staticmethod
    def has_time(df, measure):
        ''' Detects whether the given measure has time, i.e. is it measured at
            multiple timepoints during the tp or not.

            Args:
                - df (pd.DataFrame): longform master df
                - measure (str): name of measure

            Returns:
                - has_time (bool): has time?
        '''

        assert isinstance(df, pd.DataFrame)
        assert isinstance(measure, str)

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

        assert isinstance(has_time, bool)
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
