# -*- coding: utf-8 -*-

import torch
from supar.structs.dist import StructuredDistribution
from supar.structs.semiring import LogSemiring
from supar.utils.fn import stripe
from torch.distributions.utils import lazy_property


class SRLCRF(StructuredDistribution):

    def __init__(self, scores, mask=None, c_mask=None):
        super().__init__(scores)

        self.mask = mask if mask is not None else scores.new_ones(scores.shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores.new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)
        self.c_mask = c_mask

    def __add__(self, other):
        return SRLCRF(torch.stack((self.scores, other.scores), -1), self.mask, self.c_mask)

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
        s_edge = self.scores
        batch_size, seq_len = s_edge.shape[:2]
        # [seq_len, seq_len, batch_size, ...], (h->m)
        s_edge = semiring.convert(s_edge.movedim((1, 2), (1, 0)))
        s_i = semiring.zeros_like(s_edge)
        s_c = semiring.zeros_like(s_edge)
        semiring.one_(s_c.diagonal().movedim(-1, 1))
        c_mask = self.c_mask.transpose(-1, -2) if self.c_mask is not None else None

        for w in range(1, seq_len):
            n = seq_len - w

            # [n, batch_size, ...]
            il = ir = semiring.dot(stripe(s_c, n, w), stripe(s_c, n, w, (w, 1)), 1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w).copy_(semiring.mul(il, s_edge.diagonal(-w).movedim(-1, 0)).movedim(0, -1))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w).copy_(semiring.mul(ir, s_edge.diagonal(w).movedim(-1, 0)).movedim(0, -1))

            # [..., batch_size, n]
            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            c_l = semiring.dot(stripe(s_c, n, w, (0, 0), 0), stripe(s_i, n, w, (w, 0)), 1).movedim(0, -1)
            if c_mask is not None:
                c_l = semiring.zero_mask(c_l, ~c_mask.diagonal(-w, -2, -1))
            s_c.diagonal(-w).copy_(c_l)
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            c_r = semiring.dot(stripe(s_i, n, w, (0, 1)), stripe(s_c, n, w, (1, w), 0), 1).movedim(0, -1)
            if c_mask is not None:
                c_r = semiring.zero_mask(c_r, ~c_mask.diagonal(w, -2, -1))
            s_c.diagonal(w).copy_(c_r)
            s_c[0, w][self.lens.ne(w)] = semiring.zero
        return semiring.unconvert(s_c)[0][self.lens, range(batch_size)]


class SRL2oCRF(StructuredDistribution):

    def __init__(self, scores, mask=None, c_mask=None):
        super().__init__(scores)

        self.mask = mask if mask is not None else scores[0].new_ones(scores[0].shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores[0].new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)
        self.c_mask = c_mask

    def __add__(self, other):
        return SRL2oCRF([torch.stack((i, j), -1) for i, j in zip(self.scores, other.scores)], self.mask, self.c_mask)

    @lazy_property
    def argmax(self):
        return self.lens.new_zeros(self.mask.shape).masked_scatter_(self.mask,
                                                                    torch.where(self.backward(self.max.sum())[0])[2])

    def topk(self, k):
        raise NotImplementedError

    def score(self, value):
        edges, roles = value
        mask, lens = self.mask, self.lens
        s_edge, s_sib = self.scores
        prds = torch.where(edges[..., 0])[1]
        edge_mask = mask.index_fill(1, lens.new_tensor(0), 1)
        edge_mask = edge_mask.unsqueeze(1) & edge_mask.unsqueeze(2)
        edge_mask, c_mask = edge_mask & edges.gt(0), (edge_mask & edges.gt(0)).index_fill(-1, lens.new_tensor(0), 1)
        edge_mask[range(len(edges)), :, prds] = edge_mask[..., 1:].sum(-1).gt(0)
        sib_mask = edge_mask.unsqueeze(-1) & edge_mask.transpose(-1, -2).unsqueeze(1)
        sib_mask[range(len(prds)), :, prds] = ~(edge_mask & roles.ne(0).unsqueeze(-1))
        s_edge = LogSemiring.zero_mask(s_edge, ~edge_mask)
        s_sib = LogSemiring.zero_mask(s_sib, ~sib_mask)
        return self.__class__((s_edge, s_sib), mask, c_mask=c_mask).log_partition

    def forward(self, semiring):
        s_edge, s_sib = self.scores
        batch_size, seq_len = s_edge.shape[:2]
        # [seq_len, seq_len, batch_size, ...], (h->m)
        s_edge = semiring.convert(s_edge.movedim((1, 2), (1, 0)))
        # [seq_len, seq_len, seq_len, batch_size, ...], (h->m->s)
        s_sib = semiring.convert(s_sib.movedim((0, 2), (3, 0)))
        s_i = semiring.zeros_like(s_edge)
        s_s = semiring.zeros_like(s_edge)
        s_c = semiring.zeros_like(s_edge)
        semiring.one_(s_c.diagonal().movedim(-1, 1))
        c_mask = self.c_mask.transpose(-1, -2) if self.c_mask is not None else None

        for w in range(1, seq_len):
            n = seq_len - w

            # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
            #                  exp(C(j->j) + C(i->j-1)))
            #           + s(j->i)
            # [n, w, batch_size, ...]
            il = semiring.times(stripe(s_i, n, w, (w, 1)),
                                stripe(s_s, n, w, (1, 0), 0),
                                stripe(s_sib[range(w, n+w), range(n), :], n, w, (0, 1)))
            il[:, -1] = semiring.mul(stripe(s_c, n, 1, (w, w)), stripe(s_c, n, 1, (0, w - 1))).squeeze(1)
            il = semiring.sum(il, 1)
            s_i.diagonal(-w).copy_(semiring.mul(il, s_edge.diagonal(-w).movedim(-1, 0)).movedim(0, -1))
            # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
            #                  exp(C(i->i) + C(j->i+1)))
            #           + s(i->j)
            # [n, w, batch_size, ...]
            ir = semiring.times(stripe(s_i, n, w),
                                stripe(s_s, n, w, (0, w), 0),
                                stripe(s_sib[range(n), range(w, n+w), :], n, w))
            semiring.zero_(ir[0])
            ir[:, 0] = semiring.mul(stripe(s_c, n, 1), stripe(s_c, n, 1, (w, 1))).squeeze(1)
            ir = semiring.sum(ir, 1)
            s_i.diagonal(w).copy_(semiring.mul(ir, s_edge.diagonal(w).movedim(-1, 0)).movedim(0, -1))

            # [batch_size, ..., n]
            sl = sr = semiring.dot(stripe(s_c, n, w), stripe(s_c, n, w, (w, 1)), 1).movedim(0, -1)
            # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(-w).copy_(sl)
            # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(w).copy_(sr)

            # [n, batch_size, ...]
            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            c_l = semiring.dot(stripe(s_c, n, w, (0, 0), 0), stripe(s_i, n, w, (w, 0)), 1).movedim(0, -1)
            if c_mask is not None:
                c_l = semiring.zero_mask(c_l, ~c_mask.diagonal(-w, -2, -1))
            s_c.diagonal(-w).copy_(c_l)
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            c_r = semiring.dot(stripe(s_i, n, w, (0, 1)), stripe(s_c, n, w, (1, w), 0), 1).movedim(0, -1)
            if c_mask is not None:
                c_r = semiring.zero_mask(c_r, ~c_mask.diagonal(w, -2, -1))
            s_c.diagonal(w).copy_(c_r)
        return semiring.unconvert(s_c)[0][self.lens, range(batch_size)]
