#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm

from .AttentionConv import AttentionConv
from .PositionalEmbedding import LearnablePositionalEncoder

from .area_attention import AreaAttention
from .multi_head_area_attention import MultiHeadAreaAttention
from .SRGNN import GNN
# from axial_positional_embedding import AxialPositionalEmbedding

# from performer_pytorch import Performer, SelfAttention

class MHAAttention(nn.Module):
    
    def __init__(self, hidden_size, max_width, max_len, heads, dropout, aggr='max', step=1):
        super(MHAAttention, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.max_width = max_width
        self.max_len = max_len
        self.heads = heads
        self.aggr = aggr
        self.dropout = dropout
        self.feedforward = nn.Sequential(nn.Linear(self.hidden_size, 4 * self.hidden_size), nn.ReLU(), nn.Linear(4 * self.hidden_size, self.hidden_size))
        
        assert self.aggr in ['max', 'mean', 'sum', 'sample_sum', 'sample']
        assert self.max_width < self.max_len
        
        area_attention = AreaAttention(
            self.hidden_size,
            area_key_mode=self.aggr,
            max_area_height=1,
            max_area_width=self.max_width,
            memory_height=1,
            memory_width=21,
            dropout_rate=self.dropout,
            use_attn_conv=False,
            conv_layers=3,
            is_dense=False
        )
        
        self.mattn = MultiHeadAreaAttention(
            area_attention=area_attention,
            num_heads=self.heads,
            key_query_size=self.hidden_size,
            key_query_size_hidden=self.hidden_size,
            value_size=self.hidden_size,
            value_size_hidden=self.hidden_size
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def ATTNCell(self, A, hidden):
        
        attn1 = self.mattn(hidden, hidden, hidden)
        attn1 = F.layer_norm(attn1 + hidden, (self.hidden_size,))
        attn2 = self.mattn(attn1, hidden, hidden)
        attn2 = F.dropout(attn2, self.dropout, training=self.training)
        attn2 = F.layer_norm(attn1 + attn2, (self.hidden_size,))

        x = self.feedforward(attn2)
        x = F.layer_norm(attn2 + x, (self.hidden_size,))

        return x

    # def ATTNCell(self, A, hidden):
        
    #     return F.relu(self.mattn(hidden, hidden, hidden)) + F.dropout(hidden, self.dropout, training=self.training)

    def forward(self, A, hidden):
        
        for i in range(self.step):
            hidden = self.ATTNCell(A, hidden)
        
        return hidden

class SessionGraphAttn(Module):
    def __init__(self, opt, n_node):
        super(SessionGraphAttn, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.norm = opt.norm
        self.ta = opt.TA
        self.scale = opt.scale
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.res_connect = opt.res_connect
        self.heads = opt.heads
        self.conv_layers = opt.conv_layers
        self.is_dense = opt.is_dense
        self.use_attn_conv = opt.use_attn_conv
        self.use_pos = opt.use_pos
        self.softmax = opt.softmax
        self.dropout = opt.dropout
        self.step = opt.step
        self.aggr = opt.aggr
        self.last_k = opt.last_k
        self.max_width = opt.max_width
        self.dot = opt.dot
        self.conv = opt.conv
        self.area_last_conv = opt.area_last_conv
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        # self.embedding_y = nn.Embedding(self.n_node-1, self.hidden_size)
        if self.use_attn_conv:
            self.attn_conv = AttentionConv(self.heads, self.conv_layers, self.dropout, self.is_dense)
        # if self.use_pos:
        self.pe = LearnablePositionalEncoder(self.hidden_size, 146)
            # self.pe = AxialPositionalEmbedding(256, (16, 16), (128, 128))
        # self.pe = AxialPositionalEmbedding(self.hidden_size, axial_shape=(2, 73), axial_dims=(128, 128))
        self.gnn = GNN(self.hidden_size, step=opt.step)
        # self.gnn = MHAAttention(self.hidden_size, self.max_width, 11, self.heads, self.dropout, self.aggr, step=self.step)
        # self.gnn = SConv(self.hidden_size, 2, self.dropout, False)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.hidden_size, 
                self.heads,
                self.hidden_size * 4,
                self.dropout
            ), 
            self.step,
            nn.LayerNorm(self.hidden_size)
        )
        if self.area_last_conv:
            area_attention = AreaAttention(
            self.hidden_size // self.heads,
            area_key_mode=self.aggr,
            max_area_height=1,
            max_area_width=self.max_width,
            memory_height=1,
            memory_width=70,
            dropout_rate=self.dropout,
        )
            self.mattn = MultiHeadAreaAttention(
            area_attention=area_attention,
            num_heads=self.heads,
            key_query_size=self.hidden_size,
            key_query_size_hidden=self.hidden_size // self.heads,
            value_size=self.hidden_size,
            value_size_hidden=self.hidden_size // self.heads
        )
        # self.GRU = nn.GRU(self.hidden_size, self.hidden_size, 1, True, True, self.dropout)
        self.linear_qurey_zero = nn.Linear(self.last_k * self.hidden_size, self.hidden_size, bias=True)
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.gate = nn.Linear(2 * self.hidden_size, 1, bias=False)
        self.attn0 = nn.MultiheadAttention(self.hidden_size, self.heads, self.dropout)
        self.attn = nn.MultiheadAttention(self.hidden_size, self.heads, self.dropout)
        # self.attn = Performer(dim=256, depth=3, heads=8, causal=False)
        # self.attn = SelfAttention(dim=256, heads=8, causal=False)
        if self.ta:
            self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # target attention
        self.loss_function = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.normal_(std=0.1)
            

    def compute_scores(self, hidden, hidden0, mask, s):
        
        hts = [hidden[torch.arange(mask.size(0)).long(), torch.clamp(torch.sum(mask, 1)-(i+1), 0, 1000)] for i in range(self.last_k)]
        
        ht0 = hts[0].squeeze()
        if self.last_k > 1:
            ht1 = self.linear_qurey_zero(torch.cat(hts, dim=-1))
        else:
            ht1 = hts[0].squeeze()
        # mu = torch.sum(mask, dim=1, keepdim=True).unsqueeze(-1)
        # ht = torch.sum(torch.masked_fill(hidden, ~mask.unsqueeze(-1).bool(), 0), dim=1, keepdim=True).div(mu)
        ht = s.unsqueeze(1)
        # _, weights = self.attn0(hidden.permute(1, 0, 2), ht.permute(1, 0, 2), ht.permute(1, 0, 2))
        # print(weights.shape, ht.shape)
        # hidden = (1 - weights) * hidden + weights * ht.repeat(1, hidden.size(1), 1)
        hts = hidden
        ht, _ = self.attn(ht.permute(1, 0, 2), hidden.permute(1, 0, 2), hidden.permute(1, 0, 2))
        ht = ht.permute(1, 0, 2)

        sigma = torch.sigmoid(self.gate(torch.cat((hidden0, hidden), dim=-1)))
        hidden = hidden0 * sigma + hidden * (1 - sigma)

        # hidden1 = self.transformer(self.pe(hidden0).permute(1, 0, 2)).permute(1, 0, 2)

        hidden = hidden # + hidden1
        
        if self.use_pos:
            hidden = self.pe(hidden)
        if not self.area_last_conv:
            q0 = self.linear_zero(ht1).view(hts.shape[0], 1, self.hidden_size)
            q1 = self.linear_one(ht).view(hts.shape[0], 1, self.hidden_size)  # batch_size x 1 x latent_size
            q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
            assert not torch.isnan(q0).any()
            assert not torch.isnan(q1).any()
            assert not torch.isnan(q2).any()
            if self.dot:
                alpha = self.linear_three(torch.sigmoid(q1 * q2 * q0))
            else:
                alpha = self.linear_three(torch.sigmoid(q1 + q2 + q0))
            assert not torch.isnan(alpha).any()
                
            if self.softmax:
                alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
                alpha = torch.softmax(alpha, dim=1)
            assert not torch.isnan(alpha).any()
            if self.use_attn_conv:
                alpha_conv = self.attn_conv(alpha.permute(0, 2, 1)).permute(0, 2, 1)
                alpha = 0.8 * alpha + 0.2 * alpha_conv
                if self.softmax:
                    alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
                    alpha = torch.softmax(alpha, dim=1)
            a = torch.sum((alpha.unsqueeze(-1) * hidden.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        else:
            # q0 = self.linear_zero(ht0).view(hts.shape[0], 1, ht0.shape[1])
            # q1 = self.linear_one(ht).view(hts.shape[0], 1, ht0.shape[1])
            hidden = F.pad(hidden, (0, 0, 0, 70-hidden.size(1)), value=0.0)
            a = self.mattn(ht1.view(hts.shape[0], 1, self.hidden_size) * ht.view(hts.shape[0], 1, self.hidden_size), hidden, hidden)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a.squeeze(), ht0], 1))
            # a = self.linear_transform(torch.cat([a.squeeze(), ht0], 1))
        b = self.embedding.weight[1:]
        
        if self.norm:
            
            a = a.div(torch.norm(a, p=2, dim=1, keepdim=True) + 1e-12)
            
            b = b.div(torch.norm(b, p=2, dim=1, keepdim=True) + 1e-12)
            
        
        b = F.dropout(b, self.dropout, training=self.training)
        
        if self.ta:
            qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
            beta = F.softmax(b @ qt.transpose(1, 2), -1)  # batch_size x n_nodes x seq_length
            target = beta @ hidden  # batch_size x n_nodes x latent_size
            a = a.view(ht.shape [0], 1, ht.shape[1])  # b,1,d
            a = a + target  # b,n,d
            scores = torch.sum(a * b, -1)  # b,n
        else:
            scores = torch.matmul(a, b.transpose(1, 0))
        if self.scale:
            scores = 12 * scores  # 12 is the sigma factor
        return scores

    def forward(self, inputs, A, mask):
        
        # norms = torch.norm(self.embedding.weight, p=2, dim=1).data + 1e-12  # l2 norm over item embedding
        # self.embedding.weight.data = self.embedding.weight.data.div(norms.view(-1, 1).expand_as(self.embedding.weight))
            # self.embedding.weight = self.lnorm1(self.embedding.weight)
        
        hidden = self.embedding(inputs)
        
        if self.norm:
            # hidden = hidden * mask.unsqueeze(-1)
            hidden = hidden.div(torch.norm(hidden, p=2, dim=-1, keepdim=True) + 1e-12)
            # hidden[:, -1] *= 0
        
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden0 = hidden
        # if hidden.size(1) < 10:
        #     hidden = F.pad(hidden, (0, 0, 0, 10-hidden.size(1)))
       
        if self.conv:
            if self.res_connect:
                hidden = self.gnn(A, hidden) + hidden
            else:
                hidden = self.gnn(A, hidden)
        # print(hidden.shape)

        s = self.gnn.GNNPool(A, hidden)
       
        return hidden, hidden0, s
