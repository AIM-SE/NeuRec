import numpy as np
import torch
from torch.utils.data import Dataset

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def data_masks(all_usr_pois, item_tail, maxl=146):
    # add GNN+
    # cnt = 0
    # for upois in all_usr_pois:
    #     print(upois)
    #     if len(upois) >= maxl:
    #         cnt += 1
    # print(cnt)
    # all_usr_pois = [upois if len(upois) <= maxl else upois[-maxl:] for upois in all_usr_pois] 
    us_lens = [len(upois) for upois in all_usr_pois]
    # len_max = max(max(us_lens), maxl)
    # len_max = 70
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le + 1) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le + 1) for le in us_lens]
    
    return us_pois, us_msks, len_max

class Data(Dataset):
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices
    
    def __len__(self):
        
        return len(self.inputs)

    def __getitem__(self, i):

        return self.get_slice(i)

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.array(np.unique(u_input)[1:].tolist() + [0])
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input)):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] += 1
                # if u_input[i + 2] != 0:
                #     u = np.where(node == u_input[i])[0][0]
                #     v = np.where(node == u_input[i + 2])[0][0]
                #     u_A[u][v] += 0.1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        alias_inputs = torch.tensor(alias_inputs).long()
        A = torch.tensor(A).float()
        items = torch.tensor(items).long()
        mask = torch.tensor(mask).long()
        targets = torch.tensor(targets).long()
        #print(alias_inputs.shape, A.shape, items.shape, mask.shape, targets.shape)

        return alias_inputs, A, items, mask, targets

if __name__ == "__main__":
    import pickle
    data = Data(pickle.load(open("../data/yoochoose1_64/test.txt", 'rb')), shuffle=False)
    print(data.get_slice([10, 100]))