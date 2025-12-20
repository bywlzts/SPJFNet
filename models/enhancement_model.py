import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from models.base_model import BaseModel
from models.loss import CharbonnierLoss, VGGLoss, SSIM, StyleLoss, GradientLoss
from models.archs.arch_util import *
from models.archs.wavelet_normlize import *
logger = logging.getLogger('base')

class enhancement_model(BaseModel):
    def __init__(self, opt):
        super(enhancement_model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.MSELoss(reduction='sum').to(self.device)
            self.cri_pix_ill2 = nn.MSELoss(reduction='sum').to(self.device)
            self.l1_loss = nn.L1Loss().to(self.device)
            self.cri_vgg = VGGLoss()
            self.ssim_loss = SSIM()
            self.style_loss = StyleLoss()
            self.gradloss = GradientLoss()

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.nf = data['nf'].to(self.device)

        if need_GT:
            self.real_H = data['GT'].to(self.device)

        self.LL_low1, self.LL_HL1, self.LL_LH1, self.LL_HH1 = dwt_init(self.var_L)
        self.LL_low2, _, _, _ = dwt_init(self.LL_low1)
        self.LL_low3, self.LL_HL3, self.LL_LH3, self.LL_HH3 = dwt_init(self.LL_low2)
        self.LL_gt1, self.LL_HLgt1, self.LL_LHgt1, self.LL_HHgt1 = dwt_init(self.real_H)
        self.LL_gt2, _, _, _ = dwt_init(self.LL_gt1)
        self.LL_gt3, self.LL_HLgt3, self.LL_LHgt3, self.LL_HHgt3 = dwt_init(self.LL_gt2)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters_gammanet(self, step):
        """
        第一阶段：仅训练 GammaNet
        """
        for name, param in self.netG.named_parameters():
            if "gammanet" in name:
                param.requires_grad = True
            else:  # 冻结其他模块
                param.requires_grad = False

        self.optimizer_G.zero_grad()

        normalized_x_LL3, ranges_x_LL3 = normalize_wavelet_ll(self.LL_low3)
        normalized_x_LLgt3, ranges_x_LLgt3 = normalize_wavelet_ll(self.LL_gt3)
        gamma = self.netG.module.gammanet(normalized_x_LL3)

        normalized_x_LL3_gamma = torch.pow(normalized_x_LL3, gamma)  # Gamma correction
        gradient_loss = self.gradloss(normalized_x_LL3_gamma, normalized_x_LLgt3)  # 计算梯度损失

        self.log_dict['gradient_loss'] = gradient_loss.item()
        # 反向传播和优化
        gradient_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 0.01)
        self.optimizer_G.step()

    def calculate_high_freq_loss(self, preds):
        loss = 0.0
        weights = {1: 0.3, 2: 0.2, 3: 0.1}

        for i in range(1, 4):
            # 动态获取 x_H1, x_H2, x_H3 和对应的 GT
            feat_pred = preds[f'x_H{i}']
            feat_gt = preds[f'x_HGT{i}']
            # 累加 LH, HL, HH 三个方向的损失
            level_loss = sum([self.l1_loss(p, g) for p, g in zip(feat_pred, feat_gt)])
            loss += level_loss * weights[i]
        return loss * 0.5  # 总权重

    def optimize_parameters(self, step):
        for name, param in self.netG.named_parameters():
            param.requires_grad = "gammanet" not in name

        self.optimizer_G.zero_grad()
        preds = self.netG(self.var_L, gt=self.real_H)
        self.fake_H = preds['out']

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_ssim_2 = (1 - self.ssim_loss(preds['x_l2'], preds['x_LLgt2'])) * 0.1
        l_ssim_1 = (1 - self.ssim_loss(preds['x_l1'], preds['x_LLgt1'])) * 0.2
        l_ssim_0 = (1 - self.ssim_loss(self.fake_H, self.real_H)) * 0.3
        l_high_loss = self.calculate_high_freq_loss(preds)
        l_vgg = 0.0
        if self.opt['train']['vgg_loss']:
            l_vgg = self.l_pix_w * self.cri_vgg(self.fake_H, self.real_H) * 0.3
            self.log_dict['l_vgg'] = l_vgg.item()

        # 总损失
        l_final = l_pix + l_ssim_0 + l_ssim_1 + l_ssim_2 + l_high_loss + l_vgg
        l_final.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 0.01)
        self.optimizer_G.step()

        self.log_dict.update({
            'l_pix': l_pix.item(),
            'l_ssim': l_ssim_0.item(),
            'l_ssim_1': l_ssim_1.item(),
            'l_ssim_2': l_ssim_2.item(),
            'l_high_loss': l_high_loss.item()
        })

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            outputs = self.netG(self.var_L, gt=self.real_H)

            self.fake_H = outputs['out']
            self.grad_l = outputs.get('x_LL3_gamma')
            self.grad_gt = outputs.get('x_LLgt3')
            self.ssim_2_pre = outputs.get('x_l2')      # 预测的 Stage 2
            self.ssim_2_gt  = outputs.get('x_LLgt2')   # GT 的 Stage 2
            self.ssim_1_pre = outputs.get('x_l1')      # 预测的 Stage 1
            self.ssim_1_gt  = outputs.get('x_LLgt1')   # GT 的 Stage 1
            # --- 高频分量 (High Frequency) ---
            self.x_H1 = outputs.get('x_H1')
            self.x_H2 = outputs.get('x_H2')
            self.x_H3 = outputs.get('x_H3')
            self.x_HGT1 = outputs.get('x_HGT1')
            self.x_HGT2 = outputs.get('x_HGT2')
            self.x_HGT3 = outputs.get('x_HGT3')
            # --- 梯度图 ---
            self.gradient3 = outputs.get('gradient3')
            self.gradient3_gt = outputs.get('gradient3_gt')

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['x_LL3_gamma'] = self.grad_l.detach()[0].float().cpu()
        out_dict['x_LLgt3'] = self.grad_gt.detach()[0].float().cpu()
        out_dict['x_l2'] = self.ssim_2_pre.detach()[0].float().cpu()
        out_dict['x_LLgt2'] = self.ssim_2_gt.detach()[0].float().cpu()
        out_dict['x_l1'] = self.ssim_1_pre.detach()[0].float().cpu()
        out_dict['x_LLgt1'] = self.ssim_1_gt.detach()[0].float().cpu()
        out_dict['gradient3'] = self.gradient3.detach()[0].float().cpu()
        out_dict['gradient3_gt'] = self.gradient3_gt.detach()[0].float().cpu()

        del self.real_H
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
