"""
GGNN model for relation extraction.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from utils import constant, torch_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class GGNNClassifier(nn.Module):
    """ A wrapper classifier for GGNNNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.ggnn_model = GGNNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gat_model.gat.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.ggnn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output


class GGNNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.rel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt['edge_dim']) if opt['edge_dim'] > 0 else None
        self.init_embeddings()

        # state-lstm layer
        self.slstm = State_LSTM(opt, opt['hidden_dim'])
        self.slstm_drop = nn.Dropout(0.3)

        e_dim = opt['emb_dim'] + opt['ner_dim'] + opt['edge_dim']
        self.xliner = nn.Linear(e_dim, opt['hidden_dim']).cuda()

        # output mlp layers
        if opt['pool_type'] == 'entity':
            in_dim = opt['hidden_dim']*2
        elif opt['pool_type'] == 'piece':
            in_dim = opt['hidden_dim']*5
        elif opt['pool_type'] == 'all':
            in_dim = opt['hidden_dim']*6
        else:
            in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def build_masks(self, pos, maxlen):
        masks = []
        for i in pos:
            mask = [1 for _ in range(i)] + [0 for _ in range(maxlen - i)]
            masks.append(mask)
        masks = torch.tensor(masks)
        return Variable(masks.cuda()) if self.opt['cuda'] else Variable(masks)

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type, piece_pos = inputs
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        bs, N = words.size()
        rel_i_wds = torch.zeros((bs, N, N)).long()
        rel_o_wds = torch.zeros((bs, N, N)).long()
        rel_i_ner = torch.zeros((bs, N, N)).long()
        rel_o_ner = torch.zeros((bs, N, N)).long()
        rel_i_rel = torch.zeros((bs, N, N)).long()
        rel_o_rel = torch.zeros((bs, N, N)).long()
        s_i_wds = torch.zeros((bs, N, N)).long()
        s_o_wds = torch.zeros((bs, N, N)).long()
        s_o1_wds = torch.zeros((bs, N, N)).long()
        s_i_ner = torch.zeros((bs, N, N)).long()
        s_o_ner = torch.zeros((bs, N, N)).long()
        s_o1_ner = torch.zeros((bs, N, N)).long()
        s_i_rel = torch.zeros((bs, N, N)).long()
        s_o_rel = torch.zeros((bs, N, N)).long()
        s_o1_rel = torch.zeros((bs, N, N)).long()
        A_i = torch.zeros((bs, N, N))
        A_o = torch.zeros((bs, N, N))

        for i, dep in enumerate(deprel):
            for j, lb in enumerate(dep):
                if head[i][j] == 0:
                    continue
                k = head[i][j] - 1
                rel_i_wds[i][k][j] = words[i][j]
                rel_i_ner[i][k][j] = ner[i][j]
                rel_i_rel[i][k][j] = deprel[i][j]
                rel_o_wds[i][j][k] = words[i][j]
                rel_o_ner[i][j][k] = ner[i][j]
                rel_o_rel[i][j][k] = deprel[i][j]
                A_i[i][k][j] = 1
                A_o[i][j][k] = 1

        for i, t in enumerate(l):
            for j in range(t):
                rel_i_wds[i][j][j] = words[i][j]
                rel_o_wds[i][j][j] = words[i][j]
                rel_i_ner[i][j][j] = ner[i][j]
                rel_o_ner[i][j][j] = ner[i][j]
                rel_i_rel[i][j][j] = 43
                rel_o_rel[i][j][j] = 43
                A_i[i][j][j] = 1
                A_o[i][j][j] = 1
                if j == 0 or j == t - 1:
                    continue
                s_i_wds[i][j][j - 1] = words[i][j]
                s_i_wds[i][j][j + 1] = words[i][j]
                s_i_ner[i][j][j - 1] = ner[i][j]
                s_i_ner[i][j][j + 1] = ner[i][j]
                s_i_rel[i][j][j - 1] = 42
                s_i_rel[i][j][j + 1] = 42
                s_o_wds[i][j][j] = words[i][j]
                s_o_ner[i][j][j] = ner[i][j]
                s_o_rel[i][j][j] = 42
                A_i[i][j][j - 1] = 1
                A_i[i][j][j + 1] = 1
                A_o[i][j][j - 1] = 1
                A_o[i][j][j + 1] = 1
            s_i_wds[i][0][1] = words[i][0]
            s_o1_wds[i][0][0] = words[i][0]
            s_i_ner[i][0][1] = ner[i][0]
            s_o1_ner[i][0][0] = ner[i][0]
            s_i_rel[i][0][1] = 42
            s_o1_rel[i][0][0] = 42
            A_i[i][0][1] = 1
            A_o[i][0][1] = 1
            s_i_wds[i][t - 1][t - 2] = words[i][t - 1]
            s_o1_wds[i][t - 1][t - 1] = words[i][t - 1]
            s_i_ner[i][t - 1][t - 2] = ner[i][t - 1]
            s_o1_ner[i][t - 1][t - 1] = ner[i][t - 1]
            s_i_rel[i][t - 1][t - 2] = 42
            s_o1_rel[i][t - 1][t - 1] = 42
            A_i[i][t - 1][t - 2] = 1
            A_o[i][t - 1][t - 2] = 1

        e_i_rel = self.rel_emb(rel_i_rel) + self.rel_emb(s_i_rel)   # [bs, N, N, dim]
        e_i_wds = self.emb(rel_i_wds) + self.emb(s_i_wds)  # [bs, N, N, dim]
        e_i_ner = self.ner_emb(rel_i_ner) + self.ner_emb(s_i_ner)
        e_o_rel = self.rel_emb(rel_o_rel) + self.rel_emb(s_o1_rel) + 2*self.rel_emb(s_o_rel)
        e_o_wds = self.emb(rel_o_wds) + self.emb(s_o1_wds) + 2*self.emb(s_o_wds)
        e_o_ner = self.ner_emb(rel_o_ner) + self.ner_emb(s_o1_ner) + 2*self.ner_emb(s_o_ner)
        e_i = torch.cat([e_i_rel, e_i_wds, e_i_ner], dim=3)
        e_o = torch.cat([e_o_rel, e_o_wds, e_o_ner], dim=3)
        x_i = self.xliner(e_i)  # [bs, N, N, dim]
        x_o = self.xliner(e_o)
        zero = torch.zeros_like(x_i)
        A_in = A_i.unsqueeze(3).repeat(1, 1, 1, self.opt['hidden_dim'])
        A_ot = A_o.unsqueeze(3).repeat(1, 1, 1, self.opt['hidden_dim'])
        x_i = torch.where(A_in > 0, x_i, zero)
        x_o = torch.where(A_ot > 0, x_o, zero)
        x_i = x_i.sum(2)   # [bs, N, dim]
        x_o = x_o.sum(2)
        inputs = x_i, x_o, A_i, A_o

        h = self.slstm(inputs)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask

        pool_type = self.opt['pooling']
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        if self.opt['pool_type'] == 'entity':
            outputs = torch.cat([subj_out, obj_out], dim=1)
            h_out = torch.zeros(h.size())
        else:  # self.opt['pool_type'] == 'piece':
            piece1 = self.build_masks(piece_pos[:, 0], N)
            piece2 = self.build_masks(piece_pos[:, 2], N) - self.build_masks(piece_pos[:, 1] + 1, N)
            piece3 = self.build_masks(piece_pos[:, 4], N) - self.build_masks(piece_pos[:, 3] + 1, N)
            piece1_mask = piece1.eq(0).unsqueeze(2)
            piece2_mask = piece2.eq(0).unsqueeze(2)
            piece3_mask = piece3.eq(0).unsqueeze(2)
            h_p1 = pool(h, piece1_mask, type=pool_type)
            h_p2 = pool(h, piece2_mask, type=pool_type)
            h_p3 = pool(h, piece3_mask, type=pool_type)
            h_p1[h_p1 <= -1e4] = 0
            h_p2[h_p2 <= -1e4] = 0
            h_p3[h_p3 <= -1e4] = 0
            outputs = torch.cat([h_p1, h_p2, h_p3, subj_out, obj_out], dim=1)
            h_out = torch.cat([h_p1, h_p2, h_p3], dim=1)
        outputs = self.out_mlp(outputs)
        h_out = h_out.cuda() if self.opt['cuda'] else h_out.cpu()
        return outputs, h_out


class State_LSTM(nn.Module):
    """ A Contextualized State_LSTM module operated on dependency graphs. """

    def __init__(self, opt, mem_dim):
        super(State_LSTM, self).__init__()
        self.opt = opt
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.W_i1 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.W_i2 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.U_i1 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.U_i2 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.W_o1 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.W_o2 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.U_o1 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.U_o2 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.W_f1 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.W_f2 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.U_f1 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.U_f2 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.W_u1 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.W_u2 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.U_u1 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.U_u2 = nn.Parameter(torch.zeros(size=(mem_dim, mem_dim)))
        self.b_i = nn.Parameter(torch.zeros(mem_dim))
        self.b_o = nn.Parameter(torch.zeros(mem_dim))
        self.b_f = nn.Parameter(torch.zeros(mem_dim))
        self.b_u = nn.Parameter(torch.zeros(mem_dim))
        nn.init.orthogonal_(self.W_i1.data, gain=1.414)
        nn.init.orthogonal_(self.W_i2.data, gain=1.414)
        nn.init.orthogonal_(self.U_i1.data, gain=1.414)
        nn.init.orthogonal_(self.U_i2.data, gain=1.414)
        nn.init.orthogonal_(self.W_o1.data, gain=1.414)
        nn.init.orthogonal_(self.W_o2.data, gain=1.414)
        nn.init.orthogonal_(self.U_o1.data, gain=1.414)
        nn.init.orthogonal_(self.U_o2.data, gain=1.414)
        nn.init.orthogonal_(self.W_f1.data, gain=1.414)
        nn.init.orthogonal_(self.W_f2.data, gain=1.414)
        nn.init.orthogonal_(self.U_f1.data, gain=1.414)
        nn.init.orthogonal_(self.U_f2.data, gain=1.414)
        nn.init.orthogonal_(self.W_u1.data, gain=1.414)
        nn.init.orthogonal_(self.W_u2.data, gain=1.414)
        nn.init.orthogonal_(self.U_u1.data, gain=1.414)
        nn.init.orthogonal_(self.U_u2.data, gain=1.414)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        x_i, x_o, A_i, A_o = inputs  # x [bs, N, dim]  A [bs, N, N]
        bs, N, _ = x_i.size()

        h_t = torch.zeros(size=(bs, N, self.mem_dim))
        c_t = torch.zeros(size=(bs, N, self.mem_dim))

        for _ in range(self.opt['time_steps']):
            h_i = torch.bmm(A_i, h_t)
            h_o = torch.bmm(A_o, h_t)
            input = torch.bmm(x_i, self.W_i1.repeat(bs, 1, 1)) + torch.bmm(x_o, self.W_i2.repeat(bs, 1, 1)) + \
                    torch.bmm(h_i, self.U_i1.repeat(bs, 1, 1)) + torch.bmm(h_o, self.U_i2.repeat(bs, 1, 1)) + self.b_i.repeat(bs, N, 1)
            input = self.sigmoid(input)
            output = torch.bmm(x_i, self.W_o1.repeat(bs, 1, 1)) + torch.bmm(x_o, self.W_o2.repeat(bs, 1, 1)) + \
                     torch.bmm(h_i, self.U_o1.repeat(bs, 1, 1)) + torch.bmm(h_o, self.U_o2.repeat(bs, 1, 1)) + self.b_o.repeat(bs, N, 1)
            output = self.sigmoid(output)
            forget = torch.bmm(x_i, self.W_f1.repeat(bs, 1, 1)) + torch.bmm(x_o, self.W_f2.repeat(bs, 1, 1)) + \
                     torch.bmm(h_i, self.U_f1.repeat(bs, 1, 1)) + torch.bmm(h_o, self.U_f2.repeat(bs, 1, 1)) + self.b_f.repeat(bs, N, 1)
            forget = self.sigmoid(forget)
            update = torch.bmm(x_i, self.W_u1.repeat(bs, 1, 1)) + torch.bmm(x_o, self.W_u2.repeat(bs, 1, 1)) + \
                     torch.bmm(h_i, self.U_u1.repeat(bs, 1, 1)) + torch.bmm(h_o, self.U_u2.repeat(bs, 1, 1)) + self.b_u.repeat(bs, N, 1)
            update = self.sigmoid(update)
            c_t = forget * c_t + input * update
            h_t = output * self.tanh(c_t)

        return h_t


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(bs, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, bs, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0