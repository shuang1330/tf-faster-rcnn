import _init_paths
from datasets.factory import get_imdb
d = get_imdb('voc_2007_trainval')
print 'loaded'
res = d.gt_roidb()
print 'loaded the roidb'
print len(res)
for key,value in res[0].items():
    if key=='flipped':
        print key, value
    else:
        print key, value.shape
    if key=='gt_classes':
        print value
