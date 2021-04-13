import os
from glob import glob
import pandas
import numpy as np
from nilearn import input_data, image, connectome
import pennlinckit

ics_to_remove = [0,43,46,50,53,54,55,58,60,61] + list(range(64,92)) + list(range(93,100))

resids = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*/xcp/xcp_abcd/sub-*/func/*desc-residual_bold.nii.gz'))
resid = resids[pennlinckit.utils.get_sge_task_id()]
pth,fnm = os.path.split(resid)
out_pref = os.path.join(pth,'JakeBBDeriv_')

# load data and put ICA in image-space
ica = image.load_img('/cbica/projects/ISTAGING_FMRI/dropbox/melodic_IC_100.nii.gz')
img = image.load_img(resid)
icares = image.resample_to_img(ica,img)

# extract and store timeseries of ICA components
masker = input_data.NiftiMapsMasker(maps_img=icares, standardize=True,
                         memory='nilearn_cache', verbose=1)
ts = masker.fit_transform(img)
fnm = out_pref+'ts.csv'
pandas.DataFrame(ts).to_csv(fnm,index=False)

# Remove the same ICs that iStaging analysis removes 
ics = [x for x in range(100) if x not in ics_to_remove]
ts = ts[:,ics]

# Get and store derivatives
# full correlation
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
cmat = correlation_measure.fit_transform([ts])[0]
np.fill_diagonal(cmat, 0)
fnm = out_pref+'fullcorr.csv'
pandas.DataFrame(cmat).to_csv(fnm,index=False)

# partial correlation
correlation_measure = connectome.ConnectivityMeasure(kind="partial correlation")
cmat = correlation_measure.fit_transform([ts])[0]
np.fill_diagonal(cmat, 0)
fnm = out_pref+'partcorr.csv'
pandas.DataFrame(cmat).to_csv(fnm,index=False)

# nodal amplitude
namp = np.std(ts,axis=0)
fnm = out_pref+'nodeampl.csv'
pandas.DataFrame(namp).to_csv(fnm,index=False)

