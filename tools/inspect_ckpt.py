from tensorflow.python import pywrap_tensorflow
import numpy as np

file_name = '../output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(file_name)
var_to_shape_map = reader.get_variable_to_shape_map()

dic = {}
for key in sorted(var_to_shape_map):
    print key
    print reader.get_tensor(key).shape
    if key.endswith('weights'):
        print reader.get_tensor(key).shape
        dic[key] = np.sum(reader.get_tensor(key),axis=(0,1,2))
    # dic[key] = reader.get_tensor(key)
