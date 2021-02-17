import os
import math
import argparse
import numpy as np
from pytorch_lightning import profiler
import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb

from models import SessionGraphAttn
from dataset import SessionData

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]

# parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
# parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', type=bool, default=False, help='only use the global preference to predict')
parser.add_argument('--validation', type=bool, default=False, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--alpha', type=float, default=0.75, help='parameter for bera distribution')
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--TA', default=False, help='use target-aware or not')
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--res_connect', default=True, help='res connect on gnn')
parser.add_argument('--heads', type=int, default=1, help='number of attention heads')
# parser.add_argument('--conv_layers', type=int, default=1, help='number of attention conv layers.')
# parser.add_argument('--is_dense', type=int, default=1, help='use dense connection or not.')
parser.add_argument('--use_pos', default=False, action='store_true')
parser.add_argument('--use_attn_conv', default=False, action='store_true')
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--conv_layers', type=int, default=10)
parser.add_argument('--is_dense', type=bool, default=False)
parser.add_argument('--softmax', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--max_width', type=int, default=5)
parser.add_argument('--aggr', type=str, default='sum')
parser.add_argument('--dot', default=True, action='store_true')
parser.add_argument('--last_k', type=int, default=1)
parser.add_argument('--conv', default=False, action='store_true')
parser.add_argument('--area_last_conv', default=False, action='store_true')
parser.add_argument_group()

opt = parser.parse_args()
print(opt)

hyperparameter_defaults = vars(opt)
wandb.init(config=hyperparameter_defaults, project='sessRecX')
config = wandb.config
artifact = wandb.Artifact('bike-dataset', type='dataset')
class AreaAttnModel(pl.LightningModule):

    def __init__(self, opt, n_node):
        super().__init__()

        self.opt = opt

        self.best_res = [0, 0]

        self.model = SessionGraphAttn(opt, n_node)

    def forward(self, A, inputs, mask):

        return self.model(inputs, A, mask)

    def get(self, i, hidden, alias_inputs):
        return hidden[i][alias_inputs[i]]

    def training_step(self, batch, batch_idx):
        
        alias_inputs, A, items, mask, targets = batch
      
        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        targets.squeeze_()
     
        hidden, hidden0, s = self(A, items, mask)
        seq_hidden = torch.stack([self.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden0 = torch.stack([self.get(i, hidden0, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        seq_hidden0 = seq_hidden0 * mask.unsqueeze(-1)
        
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding            
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)

            seq_hidden0 = seq_hidden0.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden0, p=2, dim=-1) + 1e-12
            seq_hidden0 = seq_hidden0.div(norms.unsqueeze(-1))
            seq_hidden0 = seq_hidden0.view(seq_shape)


        scores = self.model.compute_scores(seq_hidden, seq_hidden0, mask, s)

        loss = self.model.loss_function(scores, targets - 1)
        
        wandb.log({'loss': loss})

        return loss

    def validation_step(self, batch, batch_idx):

        alias_inputs, A, items, mask, targets = batch
        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        targets.squeeze_()
        
        hidden, hidden0, s = self(A, items, mask)
        assert not torch.isnan(hidden).any()
        seq_hidden = torch.stack([self.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden0 = torch.stack([self.get(i, hidden0, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        seq_hidden0 = seq_hidden0 * mask.unsqueeze(-1)
        
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)

            seq_hidden0 = seq_hidden0.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden0, p=2, dim=-1) + 1e-12
            seq_hidden0 = seq_hidden0.div(norms.unsqueeze(-1))
            seq_hidden0 = seq_hidden0.view(seq_shape)

        scores = self.model.compute_scores(seq_hidden, seq_hidden0, mask, s)

        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(20)[1]
        # print(sub_scores.shape, targets.shape)
        sub_scores = sub_scores.cpu().detach().numpy()
        res = []
        for score, target in zip(sub_scores, targets):
            hit = float(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr = 0
            else:
                mrr = 1 / (np.where(score == target - 1)[0][0] + 1)
            res.append([hit, mrr])
        
        return torch.tensor(res)

    def validation_epoch_end(self, validation_step_outputs):
        
        output = torch.cat(validation_step_outputs, dim=0)
       
        hit = torch.mean(output[:, 0]) * 100
        mrr = torch.mean(output[:, 1]) * 100

        if hit > self.best_res[0]:
            self.best_res[0] = hit
        if mrr > self.best_res[1]:
            self.best_res[1] = mrr

        wandb.log({'hit@20': self.best_res[0], 'mrr@20': self.best_res[1]})

        self.log('hit@20', self.best_res[0])
        self.log('mrr@20', self.best_res[1])
        print(mrr, hit)

    def test_step(self, batch, idx):

        alias_inputs, A, items, mask, targets = batch
        alias_inputs.squeeze_()
        A = A.squeeze().float()
        items.squeeze_()
        mask.squeeze_()
        targets.squeeze_()
        
        # alias_inputs = torch.Tensor(alias_inputs, dtype=torch.long)
        # items = torch.Tensor(items, dtype=torch.long)
        # A = torch.Tensor(A, dtype=torch.long)
        # mask = torch.Tensor(mask, dtype=torch.long)

        hidden = self(A, items)
        seq_hidden = torch.stack([self.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=1) + 1e-12  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1).expand_as(seq_hidden))
            seq_hidden = seq_hidden.view(seq_shape)

        scores = self.model.compute_scores(seq_hidden, mask)

        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(20)[1]
        # print(sub_scores.shape, targets.shape)
        sub_scores = sub_scores.cpu().detach().numpy()
        res = []
        for score, target in zip(sub_scores, targets):
            hit = float(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr = 0
            else:
                mrr = 1 / (np.where(score == target - 1)[0][0] + 1)
            res.append([hit, mrr])
        
        return torch.tensor(res)

    def test_epoch_end(self, test_step_outputs):
        
        output = torch.cat(test_step_outputs, dim=0)
        hit = torch.mean(output[:, 0]) * 100
        mrr = torch.mean(output[:, 1]) * 100

        if hit > self.best_res[0]:
            self.best_res[0] = hit
        if mrr > self.best_res[1]:
            self.best_res[1] = mrr

        wandb.log({'hit@20': self.best_res[0], 'mrr@20': self.best_res[1]})

        self.log('hit@20', self.best_res[0])
        self.log('mrr@20', self.best_res[1])
        print(mrr, hit)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_dc_step, gamma=opt.lr_dc)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def main():

    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

    seed=123
    pl.seed_everything(seed)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    def get_freer_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return int(np.argmax(memory_available))

    session_data = SessionData(name=opt.dataset, batch_size=opt.batchSize)
    model = AreaAttnModel(opt=opt, n_node=n_node)
    early_stop_callback = EarlyStopping(
        monitor='hit@20',
        min_delta=0.00,
        patience=opt.patience,
        verbose=False,
        mode='max'
        )
# trainer = Trainer(callbacks=[early_stop_callback])
    trainer = pl.Trainer(gpus=[get_freer_gpu()], deterministic=True, max_epochs=20, callbacks=[early_stop_callback])
    # 
    trainer.fit(model, session_data)

    # trainer.test(model, session_data.test_dataloader())

if __name__ == "__main__":
    main()