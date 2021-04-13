
from glob import glob
import pennlinckit

subs = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*/xcp/xcp_abcd/sub-*/func/*desc-residual_bold.nii.gz'))

first_sub = 1
last_sub = len(subs)+1
pennlinckit.utils.submit_array_job('extract_derivatives.py',first_sub,last_sub,RAM=16)
