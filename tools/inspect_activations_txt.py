# %%
import os.path
import matplotlib.pyplot as plt
import numpy as np

# %%
path = '/home/shuang/projects/tf-faster-rcnn/activations'
num_files = len([f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))])
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
# arr_hm = np.empty([num_files,21,64], dtype=float)

arr_hm = [np.empty([num_files,21,64], dtype=float),
            np.empty([num_files,21,64], dtype=float),
            np.empty([num_files,21,128], dtype=float),
            np.empty([num_files,21,128], dtype=float),
            np.empty([num_files,21,256], dtype=float),
            np.empty([num_files,21,256], dtype=float),
            np.empty([num_files,21,256], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float),
            np.empty([num_files,21,512], dtype=float)]

# %%
for filename in os.listdir(path):
    print 'processing file {}'.format(filename)
    clas = []
    acts = []
    f = open('/'.join([path,filename]),'r')
    act_ind = 0
    for line in f.readlines():
        if line and line[0].isalpha():
            clas.append(line[:-1])
        if line.startswith('['):
            if not line.endswith(']/n'):
                acts.append([])
                acts_this_line = line[2:-1].split(' ')
                for i in acts_this_line:
                    if i is not '':
                        acts[act_ind].append(float(i))
            else:
                raise IOError('Error line with fewer numbers than expected.')
        if line.startswith(' '):
            # print 'starts with nothing'
            if line.endswith(']\n'):
                acts_this_line = line[:-2].split(' ')
                for i in acts_this_line:
                    if i is not '':
                        acts[act_ind].append(float(i))
                act_ind += 1
            else:
                acts_this_line = line.split(' ')
                for i in acts_this_line:
                    if i is not '':
                        acts[act_ind].append(float(i))

    for ind,item in enumerate(CLASSES[1:]):
        if item in clas:
            file_ind = int(filename[:-4])
            for j in range(13):
                arr_hm[j][file_ind][ind] = acts[j]

# print len(arr_hm)
# print arr_hm[1]

# %%
for i in range(11,12):
    mask = np.ma.masked_where(arr_hm[i]!=0,arr_hm[i])
    arr_hm_mask = np.ma.array(arr_hm[i],mask=mask)
    arr_hm_new = np.average(arr_hm_mask, axis=0)

    plt.imshow(arr_hm_new)
    plt.savefig('{}.png'.format(i))

# %%
fig,ax = plt.subplots()
heatmap = ax.pcolor(arr_hm_new,cmap=plt.cm.Blues,alpha=0.8)
plt.show(heatmap)
