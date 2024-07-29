import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from copy import deepcopy

from loss_utils import (count_one_to_one_pairs, dcd_loss, emd_loss, chamfer_loss, to_data, mchamfer_loss,
                        cham3DM1, cham3DM3, cham3DM5, cham3DM10, cham3DM20)

from datasets import read_h5_dataset

np.random.seed(2000)

device = torch.device("cuda")
DATA_DIR = '/shapenet16'

def get_para():
    # augment parameter
    
    x = [0.01, 0.02, 0.03, 0.04, 0.05] # point dispersion (sigma)
    y = [1.0, 0.5, 0.2, 0.1, 0.05] # density imbalance (T)

    para_arr = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    return para_arr


pts_, label = read_h5_dataset('train', 6)

para_arr = []

cd_arr3, emd_arr3 = [], []
cdp_arr3 = []

mcd1_arr3 = []
mcd1p_arr3 = []

mcd3_arr3 = []
mcd3p_arr3 = []

mcd5_arr3 = []
mcd5p_arr3 = []

mcd10_arr3 = []
mcd10p_arr3 = []

mcd20_arr3 = []
mcd20p_arr3 = []

dcd_arr3 = []

for para_idx, (disp_var, agg_var) in enumerate(tqdm(get_para())):
    cd_arr, emd_arr = [], []
    dcd_arr, mcd1_arr = [], []
    mcd3_arr, mcd5_arr = [], []
    mcd10_arr, mcd20_arr = [], []
    cdp_arr = []
    dcdp_arr, mcd1p_arr = [], []
    mcd3p_arr, mcd5p_arr = [], []
    mcd10p_arr, mcd20p_arr = [], []

    for idx, pts in enumerate(tqdm(pts_[:12000])):
        class_label = label[idx][0]
        
        pts2 = deepcopy(pts)
        pts_gt = deepcopy(pts)

        p_idx = np.random.choice(np.arange(len(pts)), 1)

        nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(pts)
        distances, indices = nbrs.kneighbors(pts)

        agg_idx = np.random.choice(np.arange(len(pts)), 10)

        for agg in agg_idx:
            agg_pts = pts[agg]  # A
            indi = indices[agg]  # A의 가까운 포인트 idx

            far_pts = pts[indi[-1]]  # A에서 가장 먼 포인트
            far_dis = np.linalg.norm(agg_pts - far_pts)

            T = agg_var

            for ind1 in indi[1:]:
                id_pts = pts[ind1]  # B
                q_abs = np.linalg.norm(id_pts - agg_pts)
                q2 = (id_pts - agg_pts)

                new_agg_pts = agg_pts + q2 * (1 - np.exp(-q_abs / T))

                pts2[ind1] = new_agg_pts

        pts2 += np.random.normal(scale=disp_var, size=pts2.shape)
        result_pts = pts2

        tp1 = to_data(np.array([pts_gt]))
        tp2 = to_data(np.array([result_pts]))

        cd, idx1_t, idx2_t = chamfer_loss(tp1, tp2)
        cd = cd.cpu().detach().numpy()
        cdp_arr.append(count_one_to_one_pairs(idx1_t, idx2_t) / 2048)
        cd_arr.append(cd)
        
        dcd, idx1_t, idx2_t = dcd_loss(tp1, tp2)
        dcd = dcd.cpu().detach().numpy()
        dcdp_arr.append(count_one_to_one_pairs(idx1_t, idx2_t) / 2048)
        dcd_arr.append(dcd)
        
        emd_v, _ = emd_loss(tp1, tp2)
        emd_v = emd_v.cpu().detach().numpy()
        emd_arr.append(emd_v)
        
        mcd1, idx1_t, idx2_t = mchamfer_loss(tp1, tp2, cham3DM1)
        mcd1 = mcd1.cpu().detach().numpy()
        mcd1p_arr.append(count_one_to_one_pairs(idx1_t, idx2_t) / 2048)
        mcd1_arr.append(mcd1)
        
        mcd3, idx1_t, idx2_t = mchamfer_loss(tp1, tp2, cham3DM3)
        mcd3 = mcd3.cpu().detach().numpy()
        mcd3p_arr.append(count_one_to_one_pairs(idx1_t, idx2_t) / 2048)
        mcd3_arr.append(mcd3)
        
        mcd5, idx1_t, idx2_t = mchamfer_loss(tp1, tp2, cham3DM5)
        mcd5 = mcd5.cpu().detach().numpy()
        mcd5p_arr.append(count_one_to_one_pairs(idx1_t, idx2_t) / 2048)
        mcd5_arr.append(mcd5)
        
        mcd10, idx1_t, idx2_t = mchamfer_loss(tp1, tp2, cham3DM10)
        mcd10 = mcd10.cpu().detach().numpy()
        mcd10p_arr.append(count_one_to_one_pairs(idx1_t, idx2_t) / 2048)
        mcd10_arr.append(mcd10)
        
        mcd20, idx1_t, idx2_t = mchamfer_loss(tp1, tp2, cham3DM20)
        mcd20 = mcd20.cpu().detach().numpy()
        mcd20p_arr.append(count_one_to_one_pairs(idx1_t, idx2_t) / 2048)
        mcd20_arr.append(mcd20)

    para_arr.append(f'{agg_var}_{disp_var}')
    
    cd_arr3.append(np.mean(cd_arr))
    cdp_arr3.append(np.mean(cdp_arr))
    
    emd_arr3.append(np.mean(emd_arr))
    dcd_arr3.append(np.mean(dcd_arr))
    
    mcd1_arr3.append(np.mean(mcd1_arr))
    mcd1p_arr3.append(np.mean(mcd1p_arr))
    
    mcd3_arr3.append(np.mean(mcd3_arr))
    mcd3p_arr3.append(np.mean(mcd3p_arr))
    
    mcd5_arr3.append(np.mean(mcd5_arr))
    mcd5p_arr3.append(np.mean(mcd5p_arr))
    
    mcd10_arr3.append(np.mean(mcd10_arr))
    mcd10p_arr3.append(np.mean(mcd10p_arr))
    
    mcd20_arr3.append(np.mean(mcd20_arr))
    mcd20p_arr3.append(np.mean(mcd20p_arr))

df = pd.DataFrame.from_dict({'param': para_arr, 'cd': cd_arr3, 'emd': emd_arr3,
                                'dcd': dcd_arr3, 'mcd1': mcd1_arr3, 'mcd3': mcd3_arr3,
                                'mcd5': mcd5_arr3, 'mcd10': mcd10_arr3, 'mcd20': mcd20_arr3,
                                'cdp': cdp_arr3, 'mcd1p': mcd1p_arr3, 'mcd3p': mcd3p_arr3,
                                'mcd5p': mcd5p_arr3, 'mcd10p': mcd10p_arr3, 'mcd20p': mcd20p_arr3})
# print(df)
df.to_csv('./error-result-MCD.csv', index=False)
