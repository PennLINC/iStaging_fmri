#import os

#os.system('source /cbica/projects/ISTAGING_FMRI/miniconda3/bin/activate xcp')

from glob import glob
import pennlinckit

subs = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*/UKB_Pipeline/*'))
first_sub = 1
last_sub = len(subs)+1
pennlinckit.utils.submit_array_job('JK_data2fmriprepbids_loop.py',first_sub,last_sub)
