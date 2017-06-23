import os.path
import numpy as np

# define th CLASSES and indices
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
SUB_CLAS = ('bicycle', 'bus', 'car','motorbike', 'person', 'train')

# define path for loading the activations_versus_classes array
hm_path = './activations_res/res.npy'

def rankmin(x):
    u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.zeros_like(counts)
    csum[1:] = counts[:-1].cumsum()
    return csum[inv]

def list_normalizer(ori_list):
    max_val = ori_list.max()
    min_val = ori_list.min()
    if max_val == 0:
        return ori_list
    normalized_list = [(i-min_val)/(max_val-min_val) for i in ori_list]
    return normalized_list

def detect_diff_one_layer(norm_hm_one_layer):
    interest_average = np.zeros((norm_hm_one_layer.shape[1],))
    diff_ind = np.zeros((norm_hm_one_layer.shape[1],))
    amplifier = 10
    for clas in SUB_CLAS:
        ind = class_to_ind[clas]
        interest_average[:] += norm_hm_one_layer[ind]
    interest_average = interest_average/len(SUB_CLAS)
    for clas in CLASSES:
        if clas not in SUB_CLAS:
            ind = class_to_ind[clas]
            temp = amplifier*(norm_hm_one_layer[ind]-interest_average)
            # print 'max: %d,min: %d'%(temp.max(),temp.min())
            temp[temp<0.5] = 0
            temp[temp>0.5] = 1
            # diff_ind[clas] = np.argsort(temp)
            diff_ind += temp
    # diff_ind = np.argsort(diff_ind)
    return diff_ind

def detect_diff_all(hm_path):
    hm_all = np.load(hm_path).item()
    norm_hm_all = {}
    hm_ind = {} # dictionary to record the diff_ind for every layer
    sub_clas_index = [class_to_ind[i] for i in SUB_CLAS]
    for key in hm_all: # for evey layer
        norm_hm_all[key] = np.zeros(hm_all[key].shape,np.float32)
        for i,sub_list in enumerate(hm_all[key]): # for every row in the layer
            norm_hm_all[key][i,:] = list_normalizer(sub_list)
        hm_ind[key] = detect_diff_one_layer(norm_hm_all[key]) # [21, 64/...]
    return hm_ind

if __name__=='__main__':
    hm_sorted = detect_diff_all(hm_path)
    for key in hm_sorted:
        print key, np.count_nonzero(hm_sorted[key])
