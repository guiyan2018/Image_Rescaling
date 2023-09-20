import logging
from collections import OrderedDict
import random
import xxlimited
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.modules.activation import ReLU6
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
#from options.options import args


from models.SRNet import SRNet as SRNet


logger = logging.getLogger('base')


class DSNetSRNetModel(BaseModel):
    def __init__(self, opt):
        super(DSNetSRNetModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        #网络DSNet SRNet
        self.netG = networks.define_G(opt).to(self.device)
        self.SRcnnNet = SRNet().to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            #开始训练
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []

            for k, v in self.SRcnnNet.named_parameters():
                #requires_grad是Pytorch中通用数据结构Tensor的一个属性，
                # 用于说明当前量是否需要在计算中保留对应的梯度信息
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            for k, v in self.netG.named_parameters():
                #requires_grad是Pytorch中通用数据结构Tensor的一个属性，
                # 用于说明当前量是否需要在计算中保留对应的梯度信息
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
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
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
    #转移数据至设备
    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = z.reshape([out.shape[0], -1])
        l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z**2) / z.shape[0]

        return l_forw_fit, l_forw_ce


    def loss_backward(self, x, y):

        x_samples=self.SRcnnNet(x=y)
        x_samples_image = x_samples[:, :3, :, :]

        #print("x_samples_.shape",x_samples.shape)
        #print("x_samples_image.shape",x_samples_image.shape)

        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

        return l_back_rec

    #优化参数
    def optimize_parameters(self, step):

        self.optimizer_G.zero_grad()

        # forward downscaling
        self.input = self.real_H
        print("input:",self.input.shape)
        #DSNnet output
        self.output = self.netG(x=self.input)

        #print("output",self.output.shape)

        zshape = self.output[:, 3:, :, :].shape

        #print("zshape:",zshape)

        LR_ref = self.ref_L.detach()

        l_forw_fit, l_forw_ce = self.loss_forward(self.output[:, :3, :, :], LR_ref, self.output[:, 3:, :, :])

        # backward upscaling
        LR = self.Quantization(self.output[:, :3, :, :])

        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1

        z=gaussian_scale * self.gaussian_batch(zshape)

        y_=LR

        channalNum=random.randint(0,8)

        y_=torch.cat((y_,z[:,channalNum:channalNum+1,:,:]),dim=1)

        l_back_rec = self.loss_backward(self.real_H, y_)

        # total loss
        loss = l_forw_fit + l_back_rec + l_forw_ce

        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_forw_ce'] = l_forw_ce.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()

    def test(self):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H
        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()

        with torch.no_grad():

            self.forw_L = self.netG(x=self.input)[:, :3, :, :]

            self.forw_L = self.Quantization(self.forw_L)

            z=gaussian_scale * self.gaussian_batch(zshape)

            y_forw=self.forw_L

            channalNum=random.randint(0,8)

            y_forw=torch.cat((y_forw,z[:,channalNum:channalNum+1,:,:]),dim=1)

            self.fake_H=self.SRcnnNet(x=y_forw)[:, :3, :, :]


        self.netG.train()

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            LR_img = self.netG(x=HR_img)[:, :3, :, :]
            LR_img = self.Quantization(self.forw_L)
        self.netG.train()

        return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
        z=gaussian_scale * self.gaussian_batch(zshape)
        y_=LR_img

        channalNum=random.randint(0,8)

        y_=torch.cat((y_,z[:,channalNum:channalNum+1,:,:]),dim=1)
        
        self.netG.eval()
        with torch.no_grad():
            HR_img = self.SRcnnNet(x=y_)[:, :3, :, :]

        self.netG.train()

        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s2, n2 = self.get_network_description(self.SRcnnNet)

        if isinstance(self.SRcnnNet, nn.DataParallel) or isinstance(self.SRcnnNet, DistributedDataParallel):
            net_struc_str2 = '{} - {}'.format(self.SRcnnNet.__class__.__name__,
                                             self.SRcnnNet.module.__class__.__name__)
        else:
            net_struc_str2 = '{}'.format(self.SRcnnNet.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network SRcnnNet structure: {}, with parameters: {:,d}'.format(net_struc_str2, n2))
            logger.info(s2)
            
        s1, n1 = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str1 = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str1 = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str1, n1))
            logger.info(s1)


    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

        load_path_D = self.opt['path']['pretrain_model_D']
        if load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.SRcnnNet, self.opt['path']['strict_load'])

    def saveAll(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.SRcnnNet, 'SRCnn', iter_label)

