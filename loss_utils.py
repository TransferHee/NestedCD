import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from ChamferDistancePytorch.Mchamfer3D1 import dist_chamfer_M3D1
from ChamferDistancePytorch.Mchamfer3D3 import dist_chamfer_M3D3
from ChamferDistancePytorch.Mchamfer3D5 import dist_chamfer_M3D5
from ChamferDistancePytorch.Mchamfer3D10 import dist_chamfer_M3D10
from ChamferDistancePytorch.Mchamfer3D20 import dist_chamfer_M3D20
from EarthMoverDistance.emd_module import emdModule

cham3D = dist_chamfer_3D.chamfer_3DDist()
mcham3D1 = dist_chamfer_M3D1.chamfer_3DDist()
mcham3D3 = dist_chamfer_M3D3.chamfer_3DDist()
mcham3D5 = dist_chamfer_M3D5.chamfer_3DDist()
mcham3D10 = dist_chamfer_M3D10.chamfer_3DDist()
mcham3D20 = dist_chamfer_M3D20.chamfer_3DDist()
emd = emdModule()

def count_one_to_one_pairs(idx1, idx2):
    # one-to-one correspondence part
    idx1_o = idx1.cpu().numpy()[0]
    idx2_o = idx2.cpu().numpy()[0]
    cnt = 0
    for i, id1 in enumerate(idx1_o):
        if i == idx2_o[id1]:
            cnt += 1
    return cnt

def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    # cham_loss = cd()
    dist1, dist2, idx1, idx2 = cham3D(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]

    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def dcd_loss(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    x = x.float().transpose(1, 2)
    gt = gt.float().transpose(1, 2)

    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (torch.mean(loss1) + torch.mean(loss2))

    return loss, idx1, idx2

def emd_loss(recons_pts, gt_pts):
    recons_pts = recons_pts.float().transpose(1, 2)
    gt_pts = gt_pts.float().transpose(1, 2)

    dis, assignment = emd(recons_pts, gt_pts, 0.05, 3000)
    loss = torch.mean(dis)

    return loss, assignment

def chamfer_loss(recons_pts, gt_pts):
    gt_pts = gt_pts.float().transpose(1, 2)
    recons_pts = recons_pts.float().transpose(1, 2)

    dist1, dist2, idx1, idx2 = cham3D(gt_pts, recons_pts)

    loss = torch.mean(dist1) + torch.mean(dist2)
    return loss, idx1, idx2

def mchamfer_loss(recons_pts, gt_pts, cham3DM):
    gt_pts = gt_pts.float().transpose(1, 2)
    recons_pts = recons_pts.float().transpose(1, 2)

    dist1, dist2, idx1, idx2 = cham3DM(gt_pts, recons_pts)

    loss = torch.mean(dist1) + torch.mean(dist2)
    return loss, idx1, idx2

def to_data(pts):
    pts = torch.from_numpy(pts)
    pts = pts.transpose(2, 1)
    return pts

def get_S_index(pts):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(pts)
    distances, _ = nbrs.kneighbors(pts)
    mean_value = np.mean(distances[1:])
    return mean_value

if __name__ == '__main__':
    print('hello world!')