import os
from glob import glob
import pennlinckit

subs = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*/fmriprep/*'))
subids = [os.path.split(x)[1] for x in subs]
fprep_dirs = [os.path.split(x)[0] for x in subs]
xcp_dirs = [os.path.join(os.path.split(x)[0],'xcp') for x in fprep_dirs]

subid = subids[pennlinckit.utils.get_sge_task_id()]
fpdir = fprep_dirs[pennlinckit.utils.get_sge_task_id()]
xcpdir = xcp_dirs[pennlinckit.utils.get_sge_task_id()]

cmd = 'singularity run -B /cbica/projects/ISTAGING_FMRI/:/home/user/data/ ~/xcpabcd.simg %s %s participant --participant_label %s --despike -p 36P  --lower-bpf 0.01  --upper-bpf 0.08'%(fpdir,xcpdir,subid)
os.system(cmd)
