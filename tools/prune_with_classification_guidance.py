import os.path
import numpy as np
import pickle

# define th CLASSES and indices
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
class_to_ind = dict(list(zip(CLASSES, list(range(len(CLASSES))))))
SUB_CLAS = ('bicycle', 'bus', 'car','motorbike', 'person', 'train')

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
            diff_ind += temp
    # diff_ind = np.argsort(diff_ind)[::-1]
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
    # define path for loading the activations_versus_classes array
    hm_path = './activations_res/res.npy'
    hm_sorted = detect_diff_all(hm_path)
    save_path = './activations_res/sorted_index.pkl'
    pickle.dump(hm_sorted,open(save_path,'wb'))
    print('Sorted index for filters are saved in %s'%save_path)
    # for key in hm_sorted:
        # print key,hm_sorted[key].shape
