#/usr/bin/python

import numpy as np
np.set_printoptions(threshold=500000000000)

f = '/mnt/lustre/xushuang2/zyfan/program/code/wav2vec2.0/fairseq/exp/results/score.npy'

a = np.load(f)
print(a)
