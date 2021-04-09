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
import matplotlib.image as mpimg
from sklearn.neighbors import KDTree

#256 316
#790 716
def index_jpg(x):
    return(int(x.split("_")[1].replace(".png", "")))
def index_label(x):
    return(int(x.split("_")[0]))
dire = "/home/jianingq/data_studio_painted_towel_r0.1/"
filename =  os.listdir(os.path.join(dire, 'camera_3'))
outeredge = [x for x in filename if x.endswith("_labels_yellow.png")]
inneredge = [x for x in filename if x.endswith("_labels_green.png")]
jpgs = [x for x in filename if x.startswith("rgb")]
outeredge = sorted(outeredge, key = index_label)
inneredge = sorted(inneredge, key = index_label)
jpgs = sorted(jpgs, key = index_jpg) 

# print(jpgs[0])
# sys.exit(1)

max_vars = []
dataset_max_var = 0
for i in range(len(outeredge)):
    print(i)
    #f, axarr = plt.subplots(1)
    #plt.axis('off')

    outer_edges = cv2.imread(os.path.join(dire, 'camera_3', outeredge[i]))
    inner_edges = cv2.imread(os.path.join(dire, 'camera_3', inneredge[i]))

    outer_edges = outer_edges[:,:,2]
    outer_edges[outer_edges<254] = 0
    outer_edges[outer_edges>=254] = 1

    inner_edges = inner_edges[:,:,2]
    inner_edges[inner_edges<254] = 1
    inner_edges[inner_edges>=254] = 0

    kernel = np.ones((3,3),np.uint8)
    inner_edges = cv2.erode(inner_edges,kernel,iterations = 1)

    im_height, im_width = np.shape(outer_edges)

    xx, yy =  np.meshgrid([x for x in range(im_width)],
                          [y for y in range(im_height)])

    temp, lbl = cv2.distanceTransformWithLabels(inner_edges.astype(np.uint8), cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # coordinates of 0-value pixels
    loc = np.where(inner_edges==0)
    xx_inner = loc[1]
    yy_inner = loc[0]
    label_to_loc = [[0,0]]

    for j in range(len(yy_inner)):
        label_to_loc.append([yy_inner[j],xx_inner[j]])

    label_to_loc = np.array(label_to_loc)

    direction = label_to_loc[lbl]

    distance = np.zeros(direction.shape)

    distance[:,:,0] = np.abs(direction[:,:,0]-yy)
    distance[:,:,1] = np.abs(direction[:,:,1]-xx)

    # Normalize distance vectors
    mag = np.linalg.norm([distance[:,:,0],distance[:,:,1]],axis = 0)+0.00001
    distance[:,:,0] = distance[:,:,0]/mag
    distance[:,:,1] = distance[:,:,1]/mag

    # Get distances of outer edges
    outer_edge_idxs = np.where(outer_edges==1)
    distance_o = distance[outer_edge_idxs]

    # Get outer edge neighbors of each outer edge point
    num_neighbour = 100

    # For every outer edge point, find its closest K neighbours 
    xx_o = xx[outer_edges==1]
    yy_o = yy[outer_edges==1]
    tree = KDTree(np.vstack([xx_o,yy_o]).T, leaf_size=2)
    dist, ind = tree.query(np.vstack([xx_o,yy_o]).T, k=num_neighbour)
    
    xx_neighbours = distance_o[ind][:,:,1]
    yy_neighbours = distance_o[ind][:,:,0]
    xx_var = np.var(xx_neighbours,axis = 1)
    yy_var = np.var(yy_neighbours,axis = 1)
    var = xx_var+yy_var

    max_var = var.max()
    if max_var > dataset_max_var:
        dataset_max_var = max_var
    max_vars.append(max_var)
    
    var_map = np.zeros((direction.shape[0], direction.shape[1]))
    var_map[outer_edge_idxs] = var

    var_path = os.path.join(dire, 'camera_3', str(outeredge[i].split("_")[0]) + "_variance.npy")
    np.save(var_path, var_map)
    print(var_path)

    # var = (var-np.min(var))/(np.max(var)-np.min(var))
    # invvar = 1/(var + 0.01)
    
    # invvar2d = np.zeros((direction.shape[0], direction.shape[1]))
    # invvar2d[outer_edge_idxs] = invvar
    #print(invvar2d.shape)
    #print(outer_edge_idxs)
    #print(np.where(invvar2d!=0))
    #print(invvar[0])
    #print(invvar[np.where(invvar!=0)[0]])

    # np.save(dire+str(outeredge[i].split("_")[0])+"_invvariance.npy", invvar2d)
    # print(dire+str(outeredge[i].split("_")[0])+"_invvariance.npy")
    #plt.imshow(invvar2d)
    #plt.colorbar()
    #plt.savefig('./{}_invvar.png'.format(i), transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi=300)

np.save(os.path.join(dire, 'metadata', 'maxvars.npy'), max_vars)
np.save(os.path.join(dire, 'metadata', 'dataset_max_var.npy'), dataset_max_var)
print("Max Variance over whole dataset: %f" % dataset_max_var)



