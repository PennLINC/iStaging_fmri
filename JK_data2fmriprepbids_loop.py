import os
from shutil import copyfile
from templateflow.api import get as get_template
from glob import glob
import nibabel as nb
import numpy as np
import pandas as pd
import tempfile
from nipype.interfaces import fsl
from nipype.interfaces.ants import Registration
#import multiprocess as mp
import pennlinckit


#bidsdir = '/cbica/projects/ISTAGING_FMRI/datasets/fmriprep/' # new directory to be like fmriprep outputs

def writejson(data,outfile):
    import json
    with open(outfile, 'w') as outfile:
        json.dump(data, outfile,sort_keys=True,indent=2)

#subid='BLSA1732' # subjectid that can be loop for all other subjects
# substr = 'BLSA_4591_01-0_10'
# jnk = substr.split('_')
# subid = jnk[0]+jnk[1]

def prep_for_fmriprep(bidsdir,rawdir,substr):
    #make subject dir, anat and func 
    subid = substr.replace('-','_').replace('_','')
    anatdir = bidsdir+'/sub-'+subid+'/anat/'
    funcdir = bidsdir+'/sub-'+subid+'/func/'
    os.makedirs(anatdir,exist_ok=True)
    os.makedirs(funcdir,exist_ok=True)


    # get t1brain and MNI template 
    t1brain=rawdir+'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/reg/highres.nii.gz'%substr
    template=str(get_template(
                'MNI152NLin2009cAsym', resolution=2, desc='brain',
                suffix='T1w', extension=['.nii', '.nii.gz']))


    ## registered T1w to template for fmriprep standard
    ### this reg files may not be used 

    tranformfile=tempfile.mkdtemp()
    reg = Registration()
    reg.inputs.fixed_image = template
    reg.inputs.moving_image = t1brain
    reg.inputs.output_transform_prefix = tranformfile+'/t12mni_'
    reg.inputs.transforms = ['Affine', 'SyN']
    reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    reg.inputs.dimension = 3
    reg.inputs.num_threads= 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.initialize_transforms_per_stage = True
    reg.inputs.metric = ['Mattes']*2
    reg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [32]*2
    reg.inputs.sampling_strategy = ['Random', None]
    reg.inputs.sampling_percentage = [0.05, None]
    reg.inputs.convergence_threshold = [1.e-8, 1.e-9]
    reg.inputs.convergence_window_size = [20]*2
    reg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
    reg.inputs.sigma_units = ['vox'] * 2
    reg.inputs.shrink_factors = [[2,1], [3,2,1]]
    reg.inputs.use_estimate_learning_rate_once = [True, True]
    reg.inputs.use_histogram_matching = [True, True] # This is the default
    #reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    reg.cmdline
    reg.run()

    ## copy transform file to fmriprep directory
    mni2twtransform = anatdir + '/sub-' + subid+'_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'
    t1w2mnitransform  =  anatdir + '/sub-' + subid+'_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    copyfile(tranformfile+'/t12mni_Composite.h5',t1w2mnitransform)
    copyfile(tranformfile+'/t12mni_InverseComposite.h5',mni2twtransform)


    ### warp the non-processed/filtered/smooth bold to fmriprep

    ### now functional 

    boldmask=rawdir +'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/mask.nii.gz'%substr
    boldref=rawdir +'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI_SBREF.nii.gz'%substr
    boldprep=rawdir +'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.nii.gz'%substr

    reffile = tempfile.mkdtemp()+'/reffile.nii.gz'
    boldstd = reffile = tempfile.mkdtemp()+'/boldstd.nii.gz'
    maskstd = reffile = tempfile.mkdtemp()+'/maskstd.nii.gz'
    aw = fsl.ApplyWarp()
    aw.inputs.in_file = boldref
    aw.inputs.ref_file = template
    aw.inputs.field_file =rawdir +'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/reg/example_func2standard_warp.nii.gz'%substr
    aw.inputs.out_file =reffile
    aw.inputs.output_type = 'NIFTI_GZ'
    res = aw.run()

    aw1 = fsl.ApplyWarp()
    aw1.inputs.interp= 'spline'
    aw1.inputs.ref_file = template
    aw1.inputs.field_file =rawdir +'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/reg/example_func2standard_warp.nii.gz'%substr
    aw1.inputs.in_file = boldprep
    aw1.inputs.out_file = boldstd
    aw1.inputs.output_type = 'NIFTI_GZ'
    res1 = aw1.run()


    aw2 = fsl.ApplyWarp()
    aw2.inputs.in_file = boldmask
    aw2.inputs.ref_file = template
    aw2.inputs.field_file =rawdir +'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/reg/example_func2standard_warp.nii.gz'%substr
    aw2.inputs.out_file = maskstd
    aw2.inputs.output_type = 'NIFTI_GZ'
    res2 = aw2.run()


    tr=nb.load(boldprep).header.get_zooms()[-1]

    jsontis={
      "RepetitionTime": np.float64(tr),
      "TaskName": 'rest',
      "SkullStripped": False,
    }

    jsmaks={ "mask":True}

    #newname
    preprocbold=funcdir + '/sub-' +subid+'_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    preprocboldjson=funcdir + '/sub-' +subid+'_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.json'
    preprocboldref=funcdir + '/sub-' +subid+'_task-rest_space-MNI152NLin2009cAsym_boldref.nii.gz'
    preprocmask=funcdir + '/sub-' +subid+'_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    preprocmaskjson=funcdir + '/sub-' +subid+'_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.json'

    copyfile(maskstd,preprocmask)
    copyfile(reffile,preprocboldref)
    copyfile(boldstd,preprocbold)
    writejson(jsontis,preprocboldjson)
    writejson(jsmaks,preprocmaskjson)

    # get wm and csf mask to extract mean signals for regressors 
    ### first warp the anatomical to bold space 
    wmask=rawdir + '/UKB_Pipeline/%s/T1/T1_fast/T1_brain_pve_2.nii.gz'%substr
    csfmask=rawdir + '/UKB_Pipeline/%s/T1/T1_fast/T1_brain_pve_0.nii.gz'%substr

    t2funcwmask=tempfile.mkdtemp()+'/wmask.nii.gz'
    t2funcwcsf=tempfile.mkdtemp()+'/csf.nii.gz'

    aw = fsl.preprocess.ApplyXFM()
    aw.inputs.in_file = wmask
    aw.inputs.reference = boldref
    aw.inputs.in_matrix_file =rawdir +'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/reg/highres2example_func.mat'%substr
    aw.inputs.out_file = t2funcwmask
    aw.inputs.apply_xfm = True
    aw.inputs.interp='nearestneighbour'
    aw.inputs.output_type = 'NIFTI_GZ'
    res = aw.run()


    aw2 = fsl.preprocess.ApplyXFM()
    aw2.inputs.in_file = csfmask
    aw2.inputs.reference = boldref
    aw2.inputs.in_matrix_file =rawdir +'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/reg/highres2example_func.mat'%substr
    aw2.inputs.out_file = t2funcwcsf
    aw2.inputs.apply_xfm = True
    aw2.inputs.interp='nearestneighbour'
    aw2.inputs.output_type = 'NIFTI_GZ'
    res2 = aw2.run()

    # binarized and extract signals
    wmbin=nb.load(t2funcwmask).get_fdata()
    wmbin[wmbin<0.99999]=0

    csfbin=nb.load(t2funcwcsf).get_fdata()
    csfbin[csfbin<0.99999]=0

    maskbin=nb.load(boldmask).get_fdata()

    bolddata=nb.load(boldprep).get_fdata()
    wm_mean=bolddata[wmbin>0,:].mean(axis=0)
    csf_mean=bolddata[csfbin>0,:].mean(axis=0)
    global_mean=bolddata[maskbin>0,:].mean(axis=0)


    #### combine all the regressors 

    mcfile=rawdir+'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/mc/prefiltered_func_data_mcf.par'%substr
    rsmdfile=rawdir+'/UKB_Pipeline/%s/fMRI_nosmooth/rfMRI.ica/mc/prefiltered_func_data_mcf_abs.rms'%substr
    motionfile=np.loadtxt(mcfile)

    rsmd=np.loadtxt(rsmdfile)
    motionparam = pd.DataFrame(motionfile,columns=['rot_x','rot_y','rot_z','trans_x','trans_y','trans_z'])

    otherparam = pd.DataFrame({'global_signal':global_mean ,'white_matter':wm_mean,'csf':csf_mean,'rmsd':rsmd })

    regressors =  pd.concat([motionparam, otherparam], axis=1)
    jsonreg = {'regressor': 'not'}
    regcsv = funcdir +'/sub-' +subid+'_task-rest_desc-confounds_timeseries.tsv'
    regjson =funcdir +'/sub-' +subid+'_task-rest_desc-confounds_timeseries.json'

    regressors.to_csv(regcsv,index=None, sep= '\t')
    writejson(jsonreg,regjson)


### SCRIPT
subs = sorted(glob('/cbica/projects/ISTAGING_FMRI/datasets/*/UKB_Pipeline/*')) # all subject directory
rawdirs = [x.split('UKB_Pipeline')[0] for x in subs]
subids = [os.path.split(x)[-1] for x in subs]
subid = subids[pennlinckit.utils.get_sge_task_id()]
rawdir = rawdirs[pennlinckit.utils.get_sge_task_id()]
bidsdir = os.path.join(rawdir,'fmriprep')
prep_for_fmriprep(bidsdir,rawdir,subid)
    # subs = os.path.join(rawdir,'UKB_Pipeline/*')
    # subids = [os.path.split(x)[-1] for x in subs]
    # for subid in subids:
    #     p = mp.Process(target=prep_for_fmriprep, 
    #                    args=(bidsdir,rawdir,subid))
    #     jobs.append(p)
    #     p.start()

