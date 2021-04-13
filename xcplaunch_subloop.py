
from glob import glob
import pennlinckit

subs = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*/fmriprep/*'))

first_sub = 1
last_sub = len(subs)+1
pennlinckit.utils.submit_array_job('XCP_subloop.py',first_sub,last_sub,RAM=16)
