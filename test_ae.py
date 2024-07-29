import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from pytorch_lightning.trainer import Trainer

from model import FullAE
from pl_datamodule import Shape16DataModule

from loss_utils import cham3D, emd

def emd_loss_d(recons_pts, gt_pts, emd_dist_arr):
    recons_pts = recons_pts.float().transpose(1, 2)
    gt_pts = gt_pts.float().transpose(1, 2)

    dis, assignment = emd(recons_pts, gt_pts, 0.05, 3000)
    d = torch.mean(dis, dim=1)

    norm_d = d * emd_dist_arr
    return norm_d, assignment

def chamfer_loss_d(recons_pts, gt_pts, cd_dis_arr):
    gt_pts = gt_pts.float().transpose(1, 2)
    recons_pts = recons_pts.float().transpose(1, 2)

    dist1, dist2, idx1, idx2 = cham3D(gt_pts, recons_pts)

    d1 = torch.mean(dist1, dim=1)
    d2 = torch.mean(dist2, dim=1)
    
    norm_d1 = d1 * cd_dis_arr
    norm_d2 = d2 * cd_dis_arr

    return norm_d1 + norm_d2, idx1, idx2

def to_point_numpy(data):
    data = torch.transpose(data, 1, 2)
    return data.cpu().detach().numpy()

def eval_model(ckpt_path, save_flag=False, save_dir='pointnet-emd-mcd5', loss_type='cd'):
    model = FullAE.load_from_checkpoint(ckpt_path)
    model.eval()

    dm = Shape16DataModule(batch_size=64)
    test_loader = dm.val_dataloader()

    device = torch.device("cuda")
    model.to(device)

    pts_arr = []
    gt_arr = []
    cd_arr = []
    cds_arr = []
    emd_arr = []
    emds_arr = []
    for pts, _, cd_dis, emd_dis, s_index in tqdm(test_loader):
        pts = pts.to(device)
        cd_dis = cd_dis.to(device)
        emd_dis = emd_dis.to(device)
        s_index = s_index.cpu().detach().numpy()
        
        if model.seg_flag:
            (recons, seg), _ = model(pts)
        else:
            recons, _ = model(pts)

        cd, _, _ = chamfer_loss_d(recons, pts, cd_dis)
        cd = cd.cpu().detach().numpy()
        cd_arr.append(cd)
        cds_arr.append(cd / s_index)

        emd, _ = emd_loss_d(recons, pts, emd_dis)
        emd = emd.cpu().detach().numpy()
        emd_arr.append(emd)
        emds_arr.append(emd / s_index)

        pts_arr.append(to_point_numpy(recons))
        gt_arr.append(to_point_numpy(pts))
        
    cd_arr = np.concatenate(cd_arr)
    emd_arr = np.concatenate(emd_arr)
    cds_arr = np.concatenate(cds_arr)
    emds_arr = np.concatenate(emds_arr)
    
    if save_flag:
        output_path = os.path.join(save_dir, loss_type)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Output & GT Point Cloud
        np.save(os.path.join(output_path, 'gt_pts.npy'), np.concatenate(gt_arr))
        np.save(os.path.join(output_path, 'pred_pts.npy'), np.concatenate(pts_arr))

        # CD & EMD Loss
        np.save(os.path.join(output_path, 'cd_err.npy'), cd_arr)
        np.save(os.path.join(output_path, 'emd_err.npy'), emd_arr)
        
        # CDS & EMDS Loss
        np.save(os.path.join(output_path, 'cds_err.npy'), cds_arr)
        np.save(os.path.join(output_path, 'emds_err.npy'), emds_arr)

    print(f'CD mean: {np.mean(cd_arr)}')
    print(f'CDs mean: {np.mean(cds_arr)}')
    print(f'EMD mean: {np.mean(emd_arr):.9f}')
    print(f'EMDs mean: {np.mean(emds_arr):.9f}')

def main():
    parser = ArgumentParser()
    parser.add_argument('--save_dir', default='./test_output/')
    parser = Shape16DataModule.add_argparse_args(parser)
    parser = FullAE.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    
    weights = './lightning_logs/version_0/checkpoints/epoch=2999-step=285000.ckpt'
    eval_model(weights, save_flag=True, save_dir=args.save_dir, loss_type='cd')

if __name__ == '__main__':
    main()
