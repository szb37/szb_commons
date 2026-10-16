'''
Shared settings of the commons codebase: allowed column values, plotting style, REDCap column
renaming, and documented examples of the measure_param format used throughout core.py.
'''

import itertools
import matplotlib.pyplot as plt

### Data settings
timings = ['bsl', 'change', 'in_dose', 'post_dose', 'post_trt']  # allowed values of the 'timing' column, see CheckDf.check_timings()
cols_checkduplicates = ['pID', 'tp', 'measure', 'time']  # columns that identify a row of df_master
cols_to_keep = ['pID', 'tp']  # REDCap columns kept by DataWrangl.get_df_measure()
corr_methods = ['pearson', 'spearman', 'kendall']  # default methods of Analysis.get_corrmats()

### Plotting settings; rcParams are applied as soon as this module is imported
plt.rcParams.update({'font.family': 'arial'})
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as editable text in SVGs instead of converting it to paths
plt.rcParams['text.usetex'] = False  # Render text with matplotlib instead of LaTeX
title_fontdict = {'fontsize': 16, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 14, 'fontweight': 'bold'}
ticklabel_fontsize = 14
save_SVG = True  # file formats used by Plots.save_fig()
save_PNG = True

### Default renaming of REDCap columns, e.g. for DataWrangl.clean_df_redcap(rename_cols=...)
redcap_cols_to_master_cols = {
    'record_id': 'pID',
    'redcap_event_name': 'tp',
    'redcap_repeat_instrument':	'repeat_instrument',
    'redcap_repeat_instance': 'repeat_instance',}


### Expected format of measure_params
#   measure_params is a list of dictionaries, where each dictionary (measure_param) defines one measure.
#   Each measure_param has the keys below; functions in core.py document which keys they use.
#       - 'instrument' (str): value of the 'instrument' column in df_master
#       - 'measure' (str): value of the 'measure' column in df_master; typically the name of the scale or subscale
#       - 'domain' (str): human-readable description of the measure, e.g. for tables and figures
#       - 'col_complete' (str or None): REDCap column that tracks completion; only rows where it equals 2 are used.
#             If None, completion is not checked (get_df_measure / get_df_vitals only)
#       - 'timing' (str): value of the 'timing' column in df_master; one of `timings` above
#       - 'tps' (list[str]): timepoints where the measure is expected; used by get_missing_scores
#       - 'is_sum_score' (bool): if True, the score is calculated from items by calc_sum_scores() and
#             stored in the REDCap column f'{measure}_LOCAL_SCORE'
#       - 'has_time' (bool): True if the measure is taken at multiple times within a timepoint (e.g. in-dose
#             measures), in which case 'time' is also needed
#       - 'do_normalize' (bool): reserved for normalizing sum scores; not used yet
#       - 'sum_score' (dict): only needed if is_sum_score is True
#             - 'col_items' (list[str]): REDCap item columns that are summed
#             - 'reverse_items' (list[str]): subset of col_items that are reverse scored
#             - 'min_score' / 'max_score' (int): min / max possible score of a single item
#   Keys needed depending on the measure:
#       - 'col_score' (str): REDCap column that stores the score; needed if is_sum_score is False
#       - 'time' (str or number): value of the 'time' column in df_master; needed if has_time is True

### Example measure_params for a sum-score measure without time
measure_params = [
    { ### MADRS - depression
        'instrument': 'MADRS',
        'measure': 'MADRS_total',
        'domain': 'MADRS - depression',
        'col_complete': 'madrs_complete',
        'timing': 'change',
        'tps': ['bsl', 'A30', 'A90', 'A365'],
        'is_sum_score': True,
        'has_time': False,
        'do_normalize': True,
        'sum_score': {
            'col_items': [f'madrs_{idx}' for idx in range(1, 11)],
            'reverse_items': [],
            'min_score': 0,
            'max_score': 6,},},]

### Example measure_params for measures with timing=in_dose, i.e. measures with time
indose_raters = ['pat', 'fac']
indose_times = ['30', '60', '90', '120', '180', '240', '300']
indose_measure_params = [
    {
        'instrument': 'INTENSITY',
        'measure': f'INTENSITY_{rater}',
        'timing': 'in_dose',
        'col_complete': 'intensity_rating_complete',
        'is_sum_score': False,
        'has_time': True,
        'col_score': f'ir_{rater}intensity{time}',
        'time': time,
    } for rater, time in itertools.product(indose_raters, indose_times)]

### Example measure_params for vitals; just like indose_measure_params, vitals have time
#   get_df_vitals() reads the two readings from columns f'{col_score}_1' and f'{col_score}_2'
vitals_measures = ['hr', 'dia', 'sys']
vitals_times = ['0', '30', '60', '90', '120', '240', '360']
vitals_measure_params = [
    {
        'instrument': 'VITALS',
        'measure': f'VITALS_{measure}',
        'timing': 'in_dose',
        'col_complete': 'vitals_record_complete',
        'is_sum_score': False,
        'has_time': True,
        'col_score': f'vs_dose{time}_{measure}',
        'time': time,
    } for time, measure in itertools.product(vitals_times, vitals_measures)]
