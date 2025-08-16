"""
Load LMDB dataset, do HessianGraphTransform, save to new LMDB dataset.
"""

import argparse
import os
import pickle
import lmdb
import torch
import copy
from tqdm import tqdm
from gadff.horm.ff_lmdb import LmdbDataset
from gadff.path_config import DATASET_DIR_HORM_EIGEN, fix_dataset_path
from ocpmodels.hessian_graph_transform import HessianGraphTransform