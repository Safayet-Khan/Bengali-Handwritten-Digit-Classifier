# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:53:05 2020

@author: safayet_khan
"""

import os
import splitfolders

RANDOM_SEED = 1337
RATIO = (0.8, 0.2)
SRC_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta_dir'
DST_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta_flow'
if not os.path.exists(path=DST_DIR):
    os.mkdir(path=DST_DIR)

splitfolders.ratio(SRC_DIR, output=DST_DIR, seed=RANDOM_SEED,
                   ratio=RATIO, group_prefix=None)
