# szb_commons

A collection of functions I frequently use across my clinical trial analysis codebases: turning
REDCap exports into long-form trial data, calculating scores, descriptive statistics,
correlations and figures.

It is shared for reproducibility purposes. You are welcome to use it at your own risk.

## Contents

| File | Description |
|---|---|
| `src/core.py` | All functions, grouped as static methods of `DataWrangl`, `Analysis`, `Plots`, `CheckDf` and `Helpers` |
| `src/config.py` | Shared settings, plotting style, and the documented format of `measure_param` dictionaries |
| `src/folders.py` | Absolute paths of the codebase folders |
| `tests/` | Tests with input and expected output fixtures |
| `commons_env.yml` | Conda environment the code is developed and tested in (Windows, Python 3.12) |

## Installation

The modules import each other as `commons_codebase.src.*`, so clone the repo into a folder named
`commons_codebase`:

```bash
git clone https://github.com/szb37/szb_commons.git commons_codebase
```

```bash
conda env create -f commons_codebase/commons_env.yml
```

Then add the folder that contains `commons_codebase` to your `PYTHONPATH` (or `sys.path`).

### Conventions

- **`df_redcap`**: a REDCap export, typically with one row per participant (`pID`) and timepoint
  (`tp`). `DataWrangl.clean_df_redcap()` and `DataWrangl.clean_pIDs()` rename and clean it.
- **`df_master`**: long-form trial data with one row per participant, timepoint, measure and time.
  Columns: `pID`, `tp`, `timing`, `instrument`, `measure`, `time` (NaN for measures without time),
  `score`, and optionally `delta_score` and `condition`. `CheckDf.check_master()` checks its
  assumptions.
- **`measure_param`**: a dictionary that defines one measure: where its scores are in the REDCap
  export, how sum scores are calculated, and when it is measured. The expected keys are documented
  in `src/config.py`, together with examples.
- **Saving outputs**: functions that accept `**save` write their result to CSV when called with
  `dir_out=...` and `fname_out=...`.

## Tests

Run from the repo root:

```bash
python -m pytest tests
```

Most tests run a function on a file from `tests/fixtures/inputs` and compare the result to the
matching file in `tests/fixtures/expected_outputs`.
