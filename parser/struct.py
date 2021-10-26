# -*- coding: utf-8 -*-

import torch
from supar.structs.dist import StructuredDistribution
from supar.structs.semiring import LogSemiring
from supar.utils.fn import stripe
from torch.distributions.utils import lazy_property


class CRFSRL(StructuredDistribution):

    def __init__(self, scores, mask=None, c_mask=None):
        super().__init__(scores, mask)

        self.mask = mask if mask is not None else scores.new_ones(scores.shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores.new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)
        self.c_mask = c_mask

    @lazy_property
    def argmax(self):
        return self.lens.new_zeros(self.mask.shape).masked_scatter_(self.mask, torch.where(self.backward(self.max.sum()))[2])

    def topk(self, k):
        raise NotImplementedError

    def score(self, value):
        mask, lens = self.mask, self.lens
        s_edge, edges = self.scores, value
        prds = torch.where(edges[..., 0])[1]
        edge_mask = mask.index_fill(1, lens.new_tensor(0), 1)
        edge_mask = edge_mask.unsqueeze(1) & edge_mask.unsqueeze(2)
        edge_mask, c_mask = edge_mask & edges.gt(0), (edge_mask & edges.gt(0)).index_fill(-1, lens.new_tensor(0), 1)
        edge_mask[range(len(edges)), :, prds] = edge_mask[..., 1:].sum(-1).gt(0)
        s_edge = LogSemiring.zero_mask(s_edge, ~edge_mask)
        return self.__class__(s_edge, mask, c_mask=c_mask).log_partition

    def forward(self, semiring):
        # [..., batch_size, seq_len, seq_len]
        s_edge = semiring.convert(self.scores.transpose(-1, -2))
        batch_size, seq_len, _ = s_edge.shape[-3:]
        s_i = semiring.zero_(torch.empty_like(s_edge))
        s_c = semiring.zero_(torch.empty_like(s_edge))
        semiring.one_(s_c.diagonal(0, -2, -1))
        c_mask = self.c_mask.transpose(-1, -2) if self.c_mask is not None else None

        for w in range(1, seq_len):
            n = seq_len - w

            # [..., batch_size, n]
            il = ir = semiring.dot(stripe(s_c, n, w), stripe(s_c, n, w, (w, 1)), -1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w, -2, -1).copy_(semiring.times(il, s_edge.diagonal(-w, -2, -1)))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w, -2, -1).copy_(semiring.times(ir, s_edge.diagonal(w, -2, -1)))

            # [..., batch_size, n]
            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            c_l = semiring.dot(stripe(s_c, n, w, (0, 0), 0), stripe(s_i, n, w, (w, 0)), -1)
            if c_mask is not None:
                c_l = semiring.zero_mask(c_l, ~c_mask.diagonal(-w, -2, -1))
            s_c.diagonal(-w, -2, -1).copy_(c_l)
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            c_r = semiring.dot(stripe(s_i, n, w, (0, 1)), stripe(s_c, n, w, (1, w), 0), -1)
            if c_mask is not None:
                c_r = semiring.zero_mask(c_r, ~c_mask.diagonal(w, -2, -1))
            s_c.diagonal(w, -2, -1).copy_(c_r)
            s_c[..., self.lens.ne(w), 0, w] = semiring.zero
        # [..., batch_size, seq_len, seq_len]
        s_c = semiring.unconvert(s_c)
        # [seq_len, batch_size, seq_len, ...]
        s_c = s_c.permute(-2, -3, -1, *range(s_c.dim() - 3))

        return s_c[0][range(batch_size), self.lens]
