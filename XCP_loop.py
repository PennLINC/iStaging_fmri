import os
from glob import glob
import pennlinckit

dss = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*'))
dss = [x for x in dss if 'fmri' not in x]
fprep_dirs = [os.path.join(x,'fmriprep') for x in dss]
xcp_dirs = [os.path.join(x,'xcp') for x in dss]

fpdir = fprep_dirs[pennlinckit.utils.get_sge_task_id()]
xcpdir = xcp_dirs[pennlinckit.utils.get_sge_task_id()]

cmd = 'singularity run -B /cbica/projects/ISTAGING_FMRI/:/home/user/data/ ~/xcpabcd.simg %s %s participant --despike -p 36P  --lower-bpf 0.01  --upper-bpf 0.08'%(fpdir,xcpdir)
os.system(cmd)
