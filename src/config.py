import itertools

### Miscs settings
measure_types = ['bsl', 'change', 'in_dose', 'post_dose', 'post_trt']
cols_checkduplicates = ['pID', 'tp', 'measure', 'time']
cols_to_keep = ['pID', 'tp']

### Examples to show format of measure_params
#   measure_params is a list of dictionaries, where each dictionary defines a measure
#   The keys below for each measure_param are mandatory, but can add other keys

### measure_params for measures with measure_type=bsl/change/post_dose/post_trt, i.e. measures without time
measure_params = [
    {
        'instrument': '11dASC',             # value for the Instrument column in df_master
        'measure': '11dASC_unity',          # value for the Measure column in df_master
        'measure_type': 'post_dose',        # value for the Type column in df_master
        'col_complete': 'dasc_complete',    # column that tracks completions in df_redcap
        'col_score': 'fivedasc_util_total', # column that contains scores in df_redcap
    },]

### example measure_params for measures with measure_type=in_dose, i.e. measures with time
raters = ['pat', 'fac']
times = ['30', '60', '90', '120', '180', '240','300']
indose_measure_params = [
    {
        'instrument': 'INTENSITY',                   # value for instrument column in df_master
        'measure': f'INTENSITY_{rater}',             # value for measure column in df_master
        'measure_type': 'in_dose',                   # value for measure_type column in df_master
        'col_complete': 'intensity_rating_complete', # completion column in df_redcap
        'col_score': f'ir_{rater}intensity{time}',   # score column in df_redcap
        'time': time,                                # value for time column in df_master
    } for rater, time in itertools.product(raters, times)]


### example measure_params for vitals; just like indose_measure_params, vitals have time
measures = ['hr', 'dia', 'sys']
times = ['0', '30', '60', '90', '120', '240', '360']
vitals_measure_params = [
    {
        'instrument': 'VITALS',
        'measure': f'VITALS_{measure}',
        'measure_type': 'in_dose',
        'col_complete': 'vitals_record_complete',
        'col_score': f'vs_dose{time}_{measure}',
        'time': time,
    } for time, measure in itertools.product(times, measures)]
