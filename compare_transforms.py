###############################################################################
#                            Acknowledgement                                 #
###############################################################################
#                                                                               #
# If you find this code useful, we kindly request that you acknowledge          #
#                                                                               #
# Author: Serge Vasylechko                                                      #
# GitHub: https://github.com/sergeicu                                           #
# Email: serge.vasylechko@tch.harvard.edu                                       #
#                                                                               #
#                                                                               #
###############################################################################

import glob 
import os 
import sys 
import math

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae




def chain_transforms(transforms):
    """Chains transforms from shape [vols][slices,parameters] to [slices,parameters]"""
    
    assert isinstance(transforms,list)
    assert transforms[0].ndim == 2
    assert transforms[0].shape[-1] == 6
    
    transforms = np.array(transforms)
    v,s,p = transforms.shape  # vols, slices, rigid params
    transforms_r = np.reshape(transforms,(v*s, p))
    return transforms_r
    
        

        
        
def apply_slicetiming_to_volume(transforms, sliceTiming, sitk_image, interpolator=sitk.sitkLinear, slice_index=0):
        
    """Applies a set of transforms according to sliceTiming order to simpleITK image object. 
    
    Returns a new object"""
    
    # create a slicetiming order 
    # Extract acquisition order index -> acqOrder
    sliceTimingIndex = list(range(0, len(sliceTiming)))
    # acqOrder is equivalent to sliceTimingIndex_sorted
    sliceTiming_sorted, acqOrder = zip(*sorted(zip(sliceTiming, sliceTimingIndex)))
        
    newimage = np.zeros(sitk.GetArrayFromImage(sitk_image).shape)
    assert newimage.ndim == 3, f"sitk_image should be a 3D image"
    for ts_i,index in zip(transforms,acqOrder):
        
        # convert to sitk.Transform format if necessary 
        if isinstance(ts_i, np.ndarray):
            #sys.exit("Please confirm transforms into a list of sitk.Transform objects first")
            assert ts_i.ndim==1 and ts_i.shape[0] ==6
            ts_i_ = sitk.Euler3DTransform()
            ts_i_.SetParameters(ts_i)
            
        else:
            ts_i_ = ts_i
            
        
        # transform image 
        i_r_t = sitk.Resample(sitk_image, sitk_image, ts_i_, interpolator, 0.0, sitk_image.GetPixelID())

        # select slice 
        image=sitk.GetArrayFromImage(i_r_t)
        
        # add slice to image
        if slice_index==0:           
            newimage[index,:, :] = image[index,:,:]
        elif slice_index==2 or slice_index==-1:
            newimage[:,:,index] = image[:,:,index]
        elif slice_index==1:
            newimage[:,index,:] = image[:,index,:]
            
    # copy file 
    new_sitk_image=sitk.GetImageFromArray(newimage)
    new_sitk_image.CopyInformation(sitk_image)
            
    return new_sitk_image
    
    
    
    

def create_slice_timing(slices,acquisition=0):
    
    """ Create fake slice timing as a list for a specific number of slices - interleaved or sequential"""
    
    
    if int(acquisition) == 0: 
        # interleaved
        order = 'interleaved'
    else:
        order = 'sequential'
    
    # create a list of floats 
    increment = 0.18
    slicetiming=[]
    timing = 0.
    for i in range(0,int(slices)):

        slicetiming.append(timing)
        timing = timing+increment
    
    if order == 'interleaved':
        ##### shuffle the timing 

        # split in half 
        half=len(slicetiming)//2
        slicetiming1=slicetiming[0:half]
        slicetiming2=slicetiming[half:]
        assert len(slicetiming) == len(slicetiming2)+len(slicetiming1)
        
        # zip two halved lists together 
        slicetiming_zipped = list(zip(slicetiming2,slicetiming1))
        
        # add to final list 
        slicetiming_final = []
        for i,j in slicetiming_zipped:
            slicetiming_final.append(i)
            slicetiming_final.append(j)
        slicetiming_final = [np.round(i,3) for i in slicetiming_final]
    else: 
        slicetiming_final = slicetiming

    return slicetiming_final
            

def construct_transform(params, angle='deg',verbose=False):
    
    """Construct euler3Dtransform from given parameters
    
    First three are rotation angles in degrees, second three are translation in mm
    """
    
    transform = sitk.Euler3DTransform()
    
    if angle=='deg':
        # turn degrees into rads
        params_ = [0, 0, 0, params[3], params[4], params[5]]
        params_[0] = math.radians(params[0])
        params_[1] = math.radians(params[1])
        params_[2] = math.radians(params[2])
        
    transform.SetParameters(params_)
    
    if verbose:
        view_transform(transform)
    
    return transform

def view_transform(transform):
    """Print values of rigid transform in mm and degrees in easily readable format"""
    
    if type(transform)==sitk.SimpleITK.Euler3DTransform:
        param=transform.GetParameters()
    
    elif isinstance(transform, np.ndarray):
        assert transform.shape[-1]==6
        if transform.ndim == 2:
            assert transform.shape[0] == 1, f"transform has shape of {transform.shape}. Shape must be [1,6]"
            param=transform[0,:]
            print("Warning")
        else:
            param=transform
    elif isinstance(transform, list):
        assert len(transform) == 6
        param=transform
    else:
        sys.exit("unknown format")
        
        
    # print verbose 
    print(f"Translation in mm  in x,y,z:\t\t\t {np.round(param[3],2)} | {np.round(param[4],2)} | {np.round(param[5],2)}")
    rad2deg=180/math.pi
    print(f"Rotation    in deg in x,y,z:\t\t\t {np.round(param[0]*rad2deg,2)} | {np.round(param[1]*rad2deg,2)} | {np.round(param[2]*rad2deg,2)}")

    
def view_transform_error(transform):
    """Print values of rigid transform in mm and degrees in easily readable format"""
    
    if type(transform)==sitk.SimpleITK.Euler3DTransform:
        param=transform.GetParameters()
    
    elif isinstance(transform, np.ndarray):
        assert transform.shape[-1]==6
        if transform.ndim == 2:
            assert transform.shape[0] == 1, f"transform has shape of {transform.shape}. Shape must be [1,6]"
            param=transform[0,:]
            print("Warning")
        else:
            param=transform
    elif isinstance(transform, list):
        assert len(transform) == 6
        param=transform
    else:
        sys.exit("unknown format")
        
        
    # print verbose 
    print(f"Translation error  in x,y,z:\t\t\t {np.round(param[3],2)} | {np.round(param[4],2)} | {np.round(param[5],2)}")
    rad2deg=180/math.pi
    print(f"Rotation    error in x,y,z:\t\t\t {np.round(param[0],2)} | {np.round(param[1],2)} | {np.round(param[2],2)}")
    
    
    
def plot_params(parameters):
    
    
    """ Plots rotation and translation parameters from a tuple that of 6 lists"""
    
    L = parameters.shape[0]
    #rot_x,rot_y,rot_z,tr_x,tr_y,tr_z = parameters
    rot_x = parameters[:,0]
    rot_y = parameters[:,1]
    rot_z = parameters[:,2]
    tr_x = parameters[:,3]
    tr_y = parameters[:,4]
    tr_z = parameters[:,5]
    
    plt.figure()
    plt.plot(list(range(0,L)), rot_x, label='x')
    plt.title('Estimated rotation (rad)')
    plt.plot(list(range(0,L)), rot_y, label='y')
    plt.plot(list(range(0,L)), rot_z, label='z')
    plt.legend()    
    
    plt.figure()
    plt.plot(list(range(0,16)), tr_x, label='x')
    plt.title('Estimated translation (mm)')
    plt.plot(list(range(0,16)), tr_y, label='y')
    plt.plot(list(range(0,16)), tr_z, label='z')
    plt.legend()
    
    
    
def plot_params_volume_list_chained(parameters_list, parameter='rx', figsize=(10,10)):
    """ Plots a parameter for multiple items (e.g. two volumes for which parameters were derived)"""
    
    # init 
    choose_params = {'rx':0,'ry':1,'rz':2, 'tx':3,'ty':4,'tz':5}
    assert parameter in choose_params.keys()
    p = choose_params[parameter]
        
    if parameter.startswith('r'):
        param_type = "rotation (rad)"
    else:
        param_type = "translation (mm)"    

    LL = len(parameters_list)

    
    # plot 
    plt.figure(figsize=figsize)
    estimates=[]
    for jj in range(0,LL):
        estimate = parameters_list[jj][:,p]
        estimates.append(estimate)
        
    # reshape 
    estimates_ = np.array(estimates)
    sh1,sh2 = estimates_.shape
    estimates__ = np.reshape(estimates_,(sh1*sh2))
        
    plt.plot(list(range(0,len(estimates__))), estimates__, label=str(jj))
    plt.title("Estimated " + param_type)
    plt.legend()    
            
def plot_params_volume_list(parameters_list, parameter='rx', labels=None,figsize=(10,10),title=None):
    """ Plots a parameter for multiple items (e.g. two volumes for which parameters were derived)"""
    
    # init 
    choose_params = {'rx':0,'ry':1,'rz':2, 'tx':3,'ty':4,'tz':5}
    assert parameter in choose_params.keys()
    p = choose_params[parameter]
        
    if parameter.startswith('r'):
        param_type = "rotation (rad)"
    else:
        param_type = "translation (mm)"    

    LL = len(parameters_list)
    L=parameters_list[0].shape[0]
    
    # plot 
    plt.figure(figsize=figsize)
    for jj in range(0,LL):
        estimate = parameters_list[jj][:,p]
        
        # choose plot label 
        if labels is None:
            label=str(jj)
        else:
            assert len(labels) == LL
            label=labels[jj]
        plt.plot(list(range(0,L)), estimate, label=label)
    if title is None:
        plt.title("Estimated " + param_type)
    else:
        plt.title(title)
    plt.legend()    
        
            
def plot_params_volume_list_simple(parameters_list, labels=None,figsize=(10,10),title=None):
    """ Plots a parameter for multiple items (e.g. two volumes for which parameters were derived)"""
    
    LL = len(parameters_list)
    L=parameters_list[0].shape[0]
    
    # plot 
    plt.figure(figsize=figsize)
    for jj in range(0,LL):
        estimate = parameters_list[jj][:]
        
        # choose plot label 
        if labels is None:
            label=str(jj)
        else:
            assert len(labels) == LL
            label=labels[jj]
        plt.plot(list(range(0,L)), estimate, label=label)
    if title is not None:
        plt.title(title)
    plt.legend()    
        
        
def get_error(parameters_list, error='mse'):
    """ Get error metrics between two arrays"""
    if error == 'mse':
        func=mse
    elif error == 'nrmse':
        func=nrmse
    elif error == 'mae':
        func=mae
    else:
        sys.exit('not implemented')
        
    assert len(parameters_list)==2
    p1,p2=parameters_list
        
    error = {'rx':0,'ry':0,'rz':0, 'tx':0,'ty':0,'tz':0}
    for i,k in enumerate(error.keys()):
        error[k] = func(p1[:,i], p2[:,i])
    return error 
        
def read_composite_transforms2(directory, prefix="svr-b0-Estimated-vol_0[0-9][0-9][0-9].tfm"):
    
    """Reads all composite transforms from a directory. Returns a list"""
    
    # basic checks 
    assert os.path.exists(directory)
    directory=directory+"/"
    
    # get files 
    tfms=glob.glob(directory+prefix)
    assert tfms
    
    # read 
    transforms=[]
    for name in tfms:        
        transform=read_composite_transform(name)
        transforms.append(transform)
    return transforms
                        
def read_composite_transforms(directory, prefix="svr-b0-Estimated-vol_0"):
    
    """Reads all composite transforms from a directory. Returns a list"""
    
    # basic checks 
    assert os.path.exists(directory)
    directory=directory+"/"
    first_tfm=directory+prefix+"000"+".tfm"
    assert os.path.exists(first_tfm), f"{first_tfm}"
    
    # get files 
    tfms=glob.glob(directory+prefix+"*"+".tfm")
    assert tfms
    
    # read 
    transforms=[]
    for i in range(0,len(tfms)):
        if i<10:
            name = directory+prefix+"00"+str(i)+".tfm"
        elif i<100:
            name = directory+prefix+"0"+str(i)+".tfm"
        else:
            name = directory+prefix+str(i)+".tfm"
        
        transform=read_composite_transform(name)
        transforms.append(transform)
    return transforms
                        
                
def read_composite_transform(file):
    """Reads composite transform from file and returns a numpy array of (L,6) shape, where L corresponds to number of transforms in a composite"""
    
    assert os.path.exists(file)
    
    #def read_composite(file):
    with open(file, 'r') as f:
        lines = f.readlines()    
    lines2=[l for l in lines if l.startswith('Parameters')]
    lines3=[l.replace('Parameters: ', '') for l in lines2]
    lines4=[l.replace('\n', '') for l in lines3]
    lines5=[l.split(' ') for l in lines4]
    lines6=[]
    for line in lines5:
        newline=[float(l) for l in line]
        lines6.append(newline)
    lines7=np.array(lines6)    
    return lines7 



def split_params(transforms):
    
    """Splits list of SimpleITK transforms into 6 separate lists of float numbers - rotationa + translation """
    
    assert isinstance(transforms,list)
    assert len(transforms[0].GetParameters())==6  
    L = len(transforms)    
    params=np.zeros((L,6))
    for i in range(0,L):
        params[i,:] = transforms[i].GetParameters()
    return params
    
    
def get_slice_transforms(rootdir,volume, basename="svr-b0-Estimated-vol_0"):
    
    """Fetches list of SimpleITK transforms for a given volume. The list represents number of slices"""
    
    basename="svr-b0-Estimated-vol_0"
    assert os.path.exists(rootdir)
    assert os.path.isdir(rootdir)
    rootdir=rootdir + "/"
    
    s1="_sl0"
    s2=".tfm"
    
    # check if first slice and first volume tfm exists 
    assert os.path.exists(rootdir+basename+ '000'+s1+'000'+s2)
    
    # check if target volume and slice exist 
    if volume<10:
        v='00'+str(volume)
    elif volume<100:
        v='0'+str(volume)
    else:
        v=str(volume)
    assert os.path.exists(rootdir+basename+ v+s1+'000'+s2)
    
    # count number of slices 
    files=glob.glob(rootdir+basename+v+s1+"*"+s2)
    assert files 
    L = len(files)
    
    # get all parameters 
    transforms = []
    for i in range(0,L):
        if i <10:
            j = '00' + str(i)
        elif i<100: 
            j = '0' + str(i)
        else:
            j = str(i)
        tfm_f=rootdir+basename+v+s1+j+s2
        assert os.path.exists(tfm_f)
        tfm = sitk.ReadTransform(tfm_f)
        transforms.append(tfm)        
    
    
    return transforms
         