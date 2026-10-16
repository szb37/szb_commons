'''
pytest configuration.

src/core.py imports its sibling modules as commons_codebase.src.*, so the folder that contains this
repo is added to sys.path; the repo folder itself must be named commons_codebase.
'''

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

# Local scratch tests are not part of the repo (see .gitignore)
collect_ignore_glob = ['private/*']
