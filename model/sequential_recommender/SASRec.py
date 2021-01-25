import numpy as np
from model.AbstractRecommender import SeqAbstractRecommender
from util import DataIterator
from util.tool import csr_to_user_dict_bytime
import torch
from util import inner_product
from util import batch_randint_choice
from util import pad_sequences
from torch.autograd import Variable


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(SeqAbstractRecommender):
    def __init__(self, dataset, conf):
        super(SASRec, self).__init__(dataset, conf)
        train_matrix, time_matrix = dataset.train_matrix, dataset.time_matrix
        self.dataset = dataset

        self.users_num, self.items_num = train_matrix.shape

        self.lr = conf["lr"]
        self.l2_emb = conf["l2_emb"]
        self.hidden_units = conf["hidden_units"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]
        self.dropout_rate = conf["dropout_rate"]
        self.max_len = conf["max_len"]
        self.num_blocks = conf["num_blocks"]
        self.num_heads = conf["num_heads"]
        self.dev = torch.device("cuda")
        self.user_pos_train = csr_to_user_dict_bytime(time_matrix, train_matrix)

        self.item_emb = torch.nn.Embedding(self.items_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(self.max_len, self.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.hidden_units,
                                                            self.num_heads,
                                                            self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def forward(self, log_seqs, pos_seqs, neg_seqs):
        # self._create_variable()
        self.seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # self.seqs = self.item_emb(log_seqs)
        self.seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        self.seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        self.seqs = self.emb_dropout(self.seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        self.seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = self.seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            self.seqs = torch.transpose(self.seqs, 0, 1)
            Q = self.attention_layernorms[i](self.seqs)
            mha_outputs, _ = self.attention_layers[i](Q, self.seqs, self.seqs,
                                                      attn_mask=attention_mask)
            self.seqs = Q + mha_outputs
            self.seqs = torch.transpose(self.seqs, 0, 1)

            self.seqs = self.forward_layernorms[i](self.seqs)
            self.seqs = self.forward_layers[i](self.seqs)
            self.seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(self.seqs)  # (U, T, C) -> (U, -1, C)
        last_emb = self.seqs[:, -1, :]

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        self.pos_logits = (log_feats * pos_embs).sum(dim=-1)
        self.neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # items_embeddings = self.item_emb.weight[:-1]  # remove the padding item
        self.all_logits = torch.matmul(last_emb, self.item_emb.weight.transpose(1,0))

        return self.pos_logits, self.neg_logits, self.all_logits



    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        train_script(self)



    def get_train_data(self):
        item_seq_list, item_pos_list, item_neg_list = [], [], []
        all_users = DataIterator(list(self.user_pos_train.keys()), batch_size=1024, shuffle=False)
        for bat_users in all_users:
            bat_seq = [self.user_pos_train[u][:-1] for u in bat_users]
            bat_pos = [self.user_pos_train[u][1:] for u in bat_users]
            n_neg_items = [len(pos) for pos in bat_pos]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion)

            # padding
            bat_seq = pad_sequences(bat_seq, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')
            bat_pos = pad_sequences(bat_pos, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')
            bat_neg = pad_sequences(bat_neg, value=self.items_num, max_len=self.max_len, padding='pre', truncating='pre')

            item_seq_list.extend(bat_seq)
            item_pos_list.extend(bat_pos)
            item_neg_list.extend(bat_neg)
        return item_seq_list, item_pos_list, item_neg_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        return predict_script(self, users, items)

def train_script(model):

    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, betas=(0.9, 0.98))

    model = model.to(model.dev)
    for epoch in range(model.epochs):
        item_seq_list, item_pos_list, item_neg_list = model.get_train_data()
        data = DataIterator(item_seq_list, item_pos_list, item_neg_list,
                                batch_size=model.batch_size, shuffle=True)
        # data = torch.tensor(data).cuda()
        for seq, pos, neg in data:
            seq, pos, neg = np.array(seq), np.array(pos), np.array(neg)

            # seq = torch.LongTensor(seq).cuda()

            # seq = torch.from_numpy(seq)
            # pos = torch.from_numpy(pos)
            # neg = torch.from_numpy(neg)
            #
            # seq = Variable(seq).cuda()
            # pos = Variable(pos).cuda()
            # neg = Variable(neg).cuda()

            pos_logits, neg_logits, all = model(seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=model.dev), torch.zeros(neg_logits.shape, device=model.dev)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += model.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
        if epoch % 100 == 0:
            result = model.evaluate_model()
            model.logger.info("epoch %d:\t%s" % (epoch, result))

def predict_script(model, users, items=None):
    users = DataIterator(users, batch_size=512, shuffle=False, drop_last=False)
    all_ratings = []
    for bat_user in users:
        bat_seq = [model.user_pos_train[u] for u in bat_user]
        bat_seq = pad_sequences(bat_seq, value=model.items_num, max_len=model.max_len, padding='pre', truncating='pre')
        bat_pos = [model.user_pos_train[u][1:] for u in bat_user]
        n_neg_items = [len(pos) for pos in bat_pos]
        exclusion = [model.user_pos_train[u] for u in bat_user]
        bat_neg = batch_randint_choice(model.items_num, n_neg_items, replace=True, exclusion=exclusion)

        bat_pos = pad_sequences(bat_pos, value=model.items_num, max_len=model.max_len, padding='pre', truncating='pre')
        bat_neg = pad_sequences(bat_neg, value=model.items_num, max_len=model.max_len, padding='pre', truncating='pre')

        _, _x, bat_ratings = model(bat_seq, bat_pos, bat_neg)
        all_ratings.extend(bat_ratings)
    all_ratings = [t.detach().cpu().numpy() for t in all_ratings]
    # all_ratings = np.array(all_ratings, dtype=np.float32)
    if items is not None:
        all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]
    return all_ratings