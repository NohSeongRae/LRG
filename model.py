import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.enc_in_dim=config['enc_in_dim']
        enc_in_dim=config['enc_in_dim']
        enc_nf=config['enc_nf']
        latent_dim=config['latent_dim']
        graph_cfg=config['graph']

        self.enc=Encoder(enc_in_dim, enc_nf, graph_cfg=graph_cfg)
        self.dec=Decoder(self.enc_content.output_channels, enc_in_dim, latent_dim=latent_dim, graph_cfg=graph_cfg)

    # def RP_trick(self, mean, var):
    #     epsilon=torch.randn(self.enc_in_dim, )
    def forward(self, xa, xb, phase='train'):
        # mean, log_var=self.enc(xa)
        # z=self.RP_trick(mean, torch.exp(0.5 *log_var))
        z=self.enc(xa)

        x=self.dec(z)
        return x




class Encoder(nn.Module):
    def __init__(self, in_channels,
                        channels,
                        graph_cfg):
        super().__init__()
        self.conv1=GraphConv(in_channels, channels) #please implement in /net/blocks
        self.gcn_mean=GraphConv(in_channels, channels)
        self.gcn_logstddev = GraphConv(in_channels, channels)
        #self.pool=NDP

    def forward(self, g, in_feat):
        h=self.conv1(g, in_feat)
        #h=self.pool(h)
        self.mean=self.gcn_mean(g,h)
        self.logstddev=self.gcn_logstddev(g,h)
        gaussian_noise=torch.randn(g.size(0),in_feat) #이 부분 확인해야함
        sampled_z=gaussian_noise*torch.exp(self.logstddev)+self.mean
        return sampled_z


class Decoder(nn.Module):
    def __init__(self, in_channels,
                        channels,
                        out_channels,
                        latent_dim,
                        graph_cfg):
        super().__init__()
        self.conv1 = GraphConv(in_channels, channels)  # please implement in /net/blocks
        #self.unpool=in_ndp
    def forward(self, z,in_channels,
                        channels,
                        out_channels,
                        latent_dim):
        h=self.conv1(z, out_channels)
        #h=unpool(h)
        h=nn.Linear(h)
        return h