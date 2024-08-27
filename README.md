This repo is a collection of functions that I frequently use during my
work across multiple codebases. It is intended for personal use. It is shared
here for reproducibility purposes: scientific codebases I maintain need to import
this repo.

To import this repo into your workflow, you will need to add to path its location:


import sys
sys.path.append({path to this repo on disk})
import codebase.src.core as commons