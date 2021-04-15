
from glob import glob
import pennlinckit

# either UKBB or XCP
version = 'UKBB'

first_sub=1
if version == 'XCP':
    subs = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*/xcp/xcp_abcd/sub-*/func/*desc-residual_bold.nii.gz'))
    last_sub = len(subs)+1
    pennlinckit.utils.submit_array_job('extract_derivatives.py',first_sub,last_sub,RAM=16)
elif version == 'UKBB':
    subs = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*/UKB_Pipeline/*/fMRI_nosmooth/rfMRI.ica/reg_standard/filtered_func_data_clean.nii.gz'))
    last_sub = len(subs)+1
    pennlinckit.utils.submit_array_job('extract_derivatives_UKBB.py',first_sub,last_sub,RAM=16)
else:
    print('Need to specify version as either UKBB or XCP')
