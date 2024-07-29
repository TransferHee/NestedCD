import torch.nn as nn
from copy import deepcopy
from argparse import ArgumentParser
from pytorch_lightning.core import LightningModule

from model_fold import FoldEncoder, FoldDecoder, FoldBiasDecoder
from model_point import PointNetfeat, BiasDecoder, FCDecoder, PointDecoder, PointGridDecoder

import torch
import torch.nn.functional as F
import torch_optimizer as opt

from loss_utils import dcd_loss, cham3D, mcham3D1, mcham3D3, mcham3D5, mcham3D10, mcham3D20, emd

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def chamfer_loss(recons_pts, gt_pts):
    gt_pts = gt_pts.float().transpose(1, 2)
    recons_pts = recons_pts.float().transpose(1, 2)

    dist1, dist2, idx1, idx2 = cham3D(gt_pts, recons_pts)

    loss = torch.mean(dist1) + torch.mean(dist2)
    return loss, idx1, idx2

def chamfer_loss_s(recons_pts, gt_pts, cd_dis_arr, s_index_arr):
    gt_pts = gt_pts.float().transpose(1, 2)
    recons_pts = recons_pts.float().transpose(1, 2)

    dist1, dist2, idx1, idx2 = cham3D(gt_pts, recons_pts)

    d1 = torch.mean(dist1, dim=1)
    d2 = torch.mean(dist2, dim=1)
    
    norm_d1 = d1 * cd_dis_arr
    norm_d2 = d2 * cd_dis_arr
    
    compensated_d1 = norm_d1 / s_index_arr
    compensated_d2 = norm_d2 / s_index_arr
    
    loss = torch.mean(compensated_d1) + torch.mean(compensated_d2)
    return loss, idx1, idx2

def mchamfer_loss(recons_pts, gt_pts, mcd, cd_dis_arr):
    gt_pts = gt_pts.float().transpose(1, 2)
    recons_pts = recons_pts.float().transpose(1, 2)

    dist1, dist2, idx1, idx2 = mcd(gt_pts, recons_pts)

    d1 = torch.mean(dist1, dim=1)
    d2 = torch.mean(dist2, dim=1)

    norm_d1 = d1 * cd_dis_arr
    norm_d2 = d2 * cd_dis_arr

    loss = torch.mean(norm_d1) + torch.mean(norm_d2)
    return loss, idx1, idx2

def mchamfer_loss_s(recons_pts, gt_pts, mcd, cd_dis_arr, s_index_arr):
    gt_pts = gt_pts.float().transpose(1, 2)
    recons_pts = recons_pts.float().transpose(1, 2)

    dist1, dist2, idx1, idx2 = mcd(gt_pts, recons_pts)

    d1 = torch.mean(dist1, dim=1)
    d2 = torch.mean(dist2, dim=1)

    norm_d1 = d1 * cd_dis_arr
    norm_d2 = d2 * cd_dis_arr

    compensated_d1 = norm_d1 / s_index_arr
    compensated_d2 = norm_d2 / s_index_arr

    loss = torch.mean(compensated_d1) + torch.mean(compensated_d2)
    return loss, idx1, idx2

def emd_loss(recons_pts, gt_pts, emd_dist_arr):
    recons_pts = recons_pts.float().transpose(1, 2)
    gt_pts = gt_pts.float().transpose(1, 2)

    dis, assignment = emd(recons_pts, gt_pts, 0.05, 3000)
    d = torch.mean(dis, dim=1)

    norm_d = d * emd_dist_arr
    loss = torch.mean(norm_d)
    
    return loss, assignment

def emd_loss_s(recons_pts, gt_pts, emd_dist_arr, s_index_arr):
    recons_pts = recons_pts.float().transpose(1, 2)
    gt_pts = gt_pts.float().transpose(1, 2)

    dis, assignment = emd(recons_pts, gt_pts, 0.05, 3000)
    d = torch.mean(dis, dim=1)

    norm_d = d * emd_dist_arr
    compensated_d = norm_d / s_index_arr
    loss = torch.mean(compensated_d)
    
    return loss, assignment

def mix_loss(recons_pts, gt_pts):
    gt_pts = gt_pts.float().transpose(1, 2)
    recons_pts = recons_pts.float().transpose(1, 2)

    dist1, dist2, idx1, idx2 = cham3D(gt_pts, recons_pts)

    cd_loss = torch.mean(dist1) + torch.mean(dist2)

    dis, assigment = emd(recons_pts, gt_pts, 0.05, 3000)
    emd_loss = torch.mean(dis)

    loss = cd_loss * 0.7 + emd_loss * 0.3

    return loss

def mse_loss(recons_pts, gt_pts):
    loss = F.mse_loss(recons_pts, gt_pts)
    return loss

def build_association():
    return nn.Sequential(
        nn.Linear(514, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512)
    )

class ReconstructionNet(nn.Module):
    def __init__(self, args, seg_class=0):
        super(ReconstructionNet, self).__init__()
        self.args = args
        model_args = args['model']

        if model_args['encoder'] == 'graph':
            self.encoder = FoldEncoder(args)
        elif model_args['encoder'] == 'point':
            self.encoder = PointNetfeat(args)

        if model_args['decoder'] == 'fold':
            self.decoder = FoldDecoder(args, seg_class)
        elif model_args['decoder'] == 'bias':
            self.decoder = BiasDecoder(args, seg_class)
        elif model_args['decoder'] == 'fc':
            self.decoder = FCDecoder(args, seg_class)
        elif model_args['decoder'] == 'point':
            self.decoder = PointDecoder(args, seg_class)
        elif model_args['decoder'] == 'bias-grid':
            self.decoder = PointGridDecoder(args, seg_class)
        elif model_args['decoder'] == 'fold-bias':
            self.decoder = FoldBiasDecoder(args, seg_class)

        print('Encoder Parameter', count_parameters(self.encoder) / 1024 / 1024, 'M')
        print('Decoder Parameter', count_parameters(self.decoder) / 1024 / 1024, 'M')

    def forward(self, input):
        global_feature, local_feature = self.encoder(input)

        if self.args['model']['decoder'] == 'point':
            output = self.decoder(global_feature, local_feature)
        else:
            output = self.decoder(global_feature)

        return output, global_feature

    def forward_glf(self, x, label):
        global_feature, local_feature = self.encoder(x)
        global_feature = torch.cat((global_feature, label), dim=1)
        output = self.decoder(global_feature)

        return output, global_feature

    def encoder_glf(self, x, label):
        global_feature, _ = self.encoder(x)
        global_feature = torch.cat((global_feature, label), dim=1)

        return global_feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

class FullAE(LightningModule):
    def __init__(
            self,
            args: dict,
            lr: float = 1e-4,
            b1: float = 0.5,
            b2: float = 0.999,
    ):

        super().__init__()

        self.save_hyperparameters()
        self.seg_flag = args['model']['segment']
        seg_class = args['parameter']['seg_class'] if self.seg_flag else 0

        self.seg_class = seg_class
        self.full_ae = ReconstructionNet(args, seg_class)
        self.args = args

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5,
                            help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999,
                            help="adam: decay of second order momentum of gradient")

        return parser

    def forward(self, pts):
        return self.full_ae(pts)

class ReconsSegAE(FullAE):
    def training_step(self, batch, batch_idx):
        full_pts, seg_label = batch
        full_pts = full_pts.float()
        seg_label = seg_label.long()

        (recons, seg), _ = self.full_ae(full_pts)

        idx2 = torch.zeros(0)

        if self.args['model']['loss'] == 'cd':
            recons_loss, idx1, idx2 = chamfer_loss(recons, full_pts)

        # reconstruction loss
        if self.args['model']['loss'] == 'mse':
            recons_loss = mse_loss(recons, full_pts)

        tqdm_dict = {'train_loss': recons_loss}
        self.log_dict(tqdm_dict)
        self.log('train_recons', recons_loss)

        # segmentation loss
        if self.args['model']['loss'] == 'mse':
            seg = seg.view(-1, self.seg_class)
            seg_label = seg_label.view(-1, 1)[:, 0]

            seg_loss = F.nll_loss(seg, seg_label)
        else:
            idx2 = idx2.long()
            reorder_seg = torch.stack([torch.index_select(ss, 0, ii) for ii, ss in zip(idx2, seg_label)], 0)

            seg = seg.view(-1, self.seg_class)
            seg_label = reorder_seg.view(-1, 1)[:, 0]

            seg_loss = F.nll_loss(seg, seg_label)

        label_eq = (torch.argmax(seg, dim=1) == seg_label).float()
        acc = torch.mean(label_eq)

        self.log('train_label', seg_loss)
        self.log('train_acc', acc)

        loss = recons_loss + 0.01 * seg_loss

        return loss

    def validation_step(self, batch, batch_index):
        full_pts, seg_label = batch
        seg_label = seg_label.long()

        (recons, seg), _ = self.full_ae(full_pts)
        recons_loss, idx1, idx2 = chamfer_loss(recons, full_pts)

        if self.args['model']['loss'] == 'cd':
            idx2 = idx2.long()
            seg_label = torch.stack([torch.index_select(ss, 0, ii) for ii, ss in zip(idx2, seg_label)], 0)

        seg = seg.view(-1, self.seg_class)
        seg_label = seg_label.view(-1, 1)[:, 0]

        seg_loss = F.nll_loss(seg, seg_label)
        loss = recons_loss + seg_loss

        label_eq = (torch.argmax(seg, dim=1) == seg_label).float()
        acc = torch.mean(label_eq)

        tqdm_dict = {'val_loss': loss}
        self.log_dict(tqdm_dict)

        self.log('val_acc', acc)
        self.log('val_label', seg_loss)
        self.log('val_cd', recons_loss)

        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt1 = opt.AdamP(
            self.full_ae.get_parameter(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            delta=0.1,
            wd_ratio=0.1
        )

        return [opt1, ], []

class ReconsAE(FullAE):
    def training_step(self, batch, batch_idx):
        full_pts, seg_label, cd_dis_arr, emd_dis_arr, s_index_arr = batch
        full_pts = full_pts.float()

        recons, _ = self.full_ae(full_pts)
        
        # reconstruction loss
        if self.args['model']['loss'] == 'mse':
            loss = mse_loss(recons, full_pts)
        elif self.args['model']['loss'] == 'mcd20':
            loss, idx1, idx2 = mchamfer_loss(recons, full_pts, mcham3D20, cd_dis_arr)
        elif self.args['model']['loss'] == 'mcd10':
            loss, idx1, idx2 = mchamfer_loss(recons, full_pts, mcham3D10, cd_dis_arr)
        elif self.args['model']['loss'] == 'mcd5':
            loss, idx1, idx2 = mchamfer_loss(recons, full_pts, mcham3D5, cd_dis_arr)
        elif self.args['model']['loss'] == 'mcd3':
            loss, idx1, idx2 = mchamfer_loss(recons, full_pts, mcham3D3, cd_dis_arr)
        elif self.args['model']['loss'] == 'mcd1':
            loss, idx1, idx2 = mchamfer_loss(recons, full_pts, mcham3D1, cd_dis_arr)
        elif self.args['model']['loss'] == 'emd':
            loss, assignment = emd_loss(recons, full_pts, emd_dis_arr)
        elif self.args['model']['loss'] == 'dcd':
            loss = dcd_loss(recons, full_pts, alpha=40, n_lambda=0.5)
        elif self.args['model']['loss'] == 'mix':
            loss = mix_loss(recons, full_pts)
        elif self.args['model']['loss'] == 'cd':
            loss, idx1, idx2 = chamfer_loss(recons, full_pts)
        
        elif self.args['model']['loss'] == 'cds':
            loss, idx1, idx2 = chamfer_loss_s(recons, full_pts, cd_dis_arr, s_index_arr)
        elif self.args['model']['loss'] == 'emds':
            loss, assignment = emd_loss_s(recons, full_pts, emd_dis_arr, s_index_arr)
        elif self.args['model']['loss'] == 'mcd1s':
            loss, idx1, idx2 = mchamfer_loss_s(recons, full_pts, mcham3D1, cd_dis_arr, s_index_arr)
        elif self.args['model']['loss'] == 'mcd3s':
            loss, idx1, idx2 = mchamfer_loss_s(recons, full_pts, mcham3D3, cd_dis_arr, s_index_arr)
        elif self.args['model']['loss'] == 'mcd5s':
            loss, idx1, idx2 = mchamfer_loss_s(recons, full_pts, mcham3D5, cd_dis_arr, s_index_arr)
        elif self.args['model']['loss'] == 'mcd10s':
            loss, idx1, idx2 = mchamfer_loss_s(recons, full_pts, mcham3D10, cd_dis_arr, s_index_arr)
        elif self.args['model']['loss'] == 'mcd20s':
            loss, idx1, idx2 = mchamfer_loss_s(recons, full_pts, mcham3D20, cd_dis_arr, s_index_arr)
        else:
            raise NotImplementedError

        tqdm_dict = {'train_loss': loss}
        
        self.log_dict(tqdm_dict)
        self.log('train_cd', loss)

        return loss

    def validation_step(self, batch, batch_index):
        full_pts, seg_label, cd_dis_arr, emd_dis_arr, s_index_arr = batch
        
        recons, _ = self.full_ae(full_pts)
        loss, _, _ = chamfer_loss(recons, full_pts)

        tqdm_dict = {'val_loss': loss}

        self.log_dict(tqdm_dict)
        self.log('val_cd', loss)

        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt1 = opt.AdamP(
            self.full_ae.get_parameter(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            delta=0.1,
            wd_ratio=0.1
        )

        return [opt1, ], []