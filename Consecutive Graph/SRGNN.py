#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

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


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 4
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.w_ih1 = Parameter(torch.Tensor(3, 2))
        self.w_hh1 = Parameter(torch.Tensor(3, 1))
        self.b_ih1 = Parameter(torch.Tensor(3))
        self.b_hh1 = Parameter(torch.Tensor(3))
        self.b_iah1 = Parameter(torch.Tensor(1))
        self.b_oah1 = Parameter(torch.Tensor(1))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_in_1 = nn.Linear(self.hidden_size, 1, bias=True)
        self.linear_edge_out_1 = nn.Linear(self.hidden_size, 1, bias=True)
        self.linear_edge_in_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(2, 1, bias=True)
        self.attn = nn.MultiheadAttention(self.hidden_size, 8, 0.1)

    def GNNPool(self, A, hidden):

        hidden0 = hidden

        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in_1(hidden)) + self.b_iah1
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out_1(hidden)) + self.b_oah1
        hidden = self.attn(hidden.permute(1, 0, 2), hidden.permute(1, 0, 2), hidden.permute(1, 0, 2))[1]
        hidden = torch.sum(hidden, -1, keepdim=True)
        inputs = torch.cat([input_in, input_out], 2)
        hy = self.linear_edge_f(inputs)
        gi = F.linear(inputs, self.w_ih1, self.b_ih1)
        gh = F.linear(hidden, self.w_hh1, self.b_hh1)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        
        hy = torch.softmax(hy, dim=-1)

        return torch.sum(hy * hidden0, dim=1)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # input_in_1 = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in_1(input_in))
        # input_out_1 = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out_2(input_out))
        # input_in_2 = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in_2(input_in))
        # input_out_2 = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out_2(input_out))
        # inputs = torch.cat([input_in, input_out, input_in_1, input_out_1], 2)
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        # input_in_1 = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in_1(input_in))
        # input_out_1 = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out_2(input_out))
        # inputs = torch.cat([input_in_1, input_out_1], 2)
        # gi = F.linear(inputs, self.w_ih, self.b_ih)
        # gh = F.linear(hidden, self.w_hh, self.b_hh)
        # i_r, i_i, i_n = gi.chunk(3, 2)
        # h_r, h_i, h_n = gh.chunk(3, 2)
        # resetgate = torch.sigmoid(i_r + h_r)
        # inputgate = torch.sigmoid(i_i + h_i)
        # newgate = torch.tanh(i_n + resetgate * h_n)
        # hy1 = newgate + inputgate * (hidden - newgate)

        # return self.linear_edge_f(torch.cat([hy, hy1], dim=-1))

        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
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
        
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        if self.use_attn_conv:
            self.attn_conv = AttentionConv(self.heads, self.conv_layers, self.is_dense)
        if self.use_pos:
            self.pe = LearnablePositionalEncoder(self.hidden_size, 10)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        if self.ta:
            self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        # ht1 = hidden[torch.arange(mask.shape[0]).long(), torch.clamp(torch.sum(mask, 1) - 2, 0, 1000)]
        # ht2 = hidden[torch.arange(mask.shape[0]).long(), torch.clamp(torch.sum(mask, 1) - 3, 0, 1000)]
        # ht3 = hidden[torch.arange(mask.shape[0]).long(), torch.clamp(torch.sum(mask, 1) - 4, 0, 1000)]
        # ht = torch.cat((ht0, ht1, ht2, ht3), dim=-1)
        # ht = ht0
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 * q2))
       
            
        a = torch.sum((alpha.unsqueeze(-1) * hidden.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        # if self.norm:
        #     norms = torch.norm(a, p=2, dim=1, keepdim=True)  # a needs to be normalized too
        #     a = a.div(norms)
        #     norms = torch.norm(self.embedding.weight, p=2, dim=1).data  # l2 norm over item embedding again for b
        #     self.embedding.weight.data = self.embedding.weight.data.div(norms.view(-1, 1).expand_as(self.embedding.weight))
            # a = self.lnorm(a)
            # self.embedding.weight.data = self.lnorm(self.embedding.weight.data)
        
        b = F.dropout(self.embedding.weight[1:], self.dropout, training=self.training)  # n_nodes x latent_size
        if self.ta:
            qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
            beta = F.softmax(b @ qt.transpose(1, 2), -1)  # batch_size x n_nodes x seq_length
            target = beta @ hidden  # batch_size x n_nodes x latent_size
            a = a.view(ht.shape[0], 1, ht.shape[1])  # b,1,d
            a = a + target  # b,n,d
            scores = torch.sum(a * b, -1)  # b,n
        else:
            scores = torch.matmul(a, b.transpose(1, 0))
        if self.scale:
            scores = 12 * scores  # 16 is the sigma factor
        return scores

    def forward(self, inputs, A):
        # if self.norm:
        #     norms = torch.norm(self.embedding.weight, p=2, dim=1).data  # l2 norm over item embedding
        #     self.embedding.weight.data = self.embedding.weight.data.div(norms.view(-1, 1).expand_as(self.embedding.weight))
        hidden = F.dropout(self.embedding(inputs), self.dropout, training=self.training)
        hidden = self.gnn(A, hidden)
        return hidden