import sys
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from skimage import morphology
import os
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def index_label(x):
    return int(x.split("_")[0])
dire = "/home/jianingq/data_studio_painted_towel_r0.1/"
filename =  os.listdir(os.path.join(dire, 'camera_3'))
variances = [x for x in filename if x.endswith("_variance.npy") and 'inverse' not in x]
variances = sorted(variances, key=index_label)
print(len(variances))

dataset_max_var = np.load(os.path.join(dire, 'metadata', 'dataset_max_var.npy'))
print("Dataset max variance: %f" % dataset_max_var)

for varpath in variances:
    # print(i)

    varmap = np.load(os.path.join(dire, 'camera_3', varpath))

    scaled_varmap = varmap / dataset_max_var
    inverse_varmap = 1 - scaled_varmap
    i = varpath.split("_")[0]
    np.save(os.path.join(dire, 'camera_3', "%s_inverse_variance.npy" % i), inverse_varmap)
