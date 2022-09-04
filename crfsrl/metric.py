# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
from supar.utils.common import CACHE
from supar.utils.fn import download
from supar.utils.metric import Metric


class SpanSRLMetric(Metric):

    URL = 'http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz'
    PATH = os.path.join(CACHE, 'data', 'srl')

    def __init__(
            self,
            loss: Optional[float] = None,
            preds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            golds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            eps=1e-12
    ) -> SpanSRLMetric:
        super().__init__(eps=eps)

        self.prd_tp = 0
        self.prd_pred = 0
        self.prd_gold = 0
        self.prd_cm = 0
        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0

        self.script = os.path.join(self.PATH, 'srlconll-1.1/bin/srl-eval.pl')
        if not os.path.exists(self.script):
            download(self.URL, self.PATH)

        if loss is not None:
            self(loss, preds, golds)

    def __repr__(self):
        s = f"loss: {self.loss:.4f} - "
        s += f"PRD: {self.prd_f:6.2%} CM: {self.cm:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"
        return s

    def __call__(
        self,
        loss: float,
        preds: List,
        golds: List,
    ) -> SpanSRLMetric:
        lens = [max(max([*i, *j], key=lambda x: max(x[:3]))[:3]) if i or j else 1 for i, j in zip(preds, golds)]
        ftemp = tempfile.mkdtemp()
        fpred, fgold = os.path.join(ftemp, 'pred'), os.path.join(ftemp, 'gold')
        with open(fpred, 'w') as f:
            f.write('\n\n'.join([self.span2prop(spans, lens[i]) for i, spans in enumerate(preds)]))
        with open(fgold, 'w') as f:
            f.write('\n\n'.join([self.span2prop(spans, lens[i]) for i, spans in enumerate(golds)]))
        os.environ['PERL5LIB'] = os.path.join(self.PATH, 'srlconll-1.1', 'lib:$PERL5LIB')
        p_out = subprocess.check_output(['perl', f'{self.script}', f'{fpred}', f'{fgold}'], stderr=subprocess.STDOUT).decode()
        r_out = subprocess.check_output(['perl', f'{self.script}', f'{fgold}', f'{fpred}'], stderr=subprocess.STDOUT).decode()
        p_out = [i for i in p_out.split('\n') if 'Overall' in i][0].split()
        r_out = [i for i in r_out.split('\n') if 'Overall' in i][0].split()
        shutil.rmtree(ftemp)

        self.n += len(preds)
        self.count += 1
        self.total_loss += float(loss)
        self.tp += int(p_out[1])
        self.pred += int(p_out[3]) + int(p_out[1])
        self.gold += int(r_out[3]) + int(p_out[1])
        for pred, gold in zip(preds, golds):
            pred_props, gold_props = defaultdict(list), defaultdict(list)
            for span in pred:
                pred_props[span[0]].append(tuple(span[1:]))
            for span in gold:
                gold_props[span[0]].append(tuple(span[1:]))
            self.prd_tp += len(pred_props.keys() & gold_props.keys())
            self.prd_pred += len(pred_props)
            self.prd_gold += len(gold_props)
            self.prd_cm += sum(sorted(args) == sorted(pred_props[prd]) for prd, args in gold_props.items())
        return self

    def __add__(self, other: SpanSRLMetric) -> SpanSRLMetric:
        metric = SpanSRLMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss

        metric.prd_tp = self.prd_tp + other.prd_tp
        metric.prd_pred = self.prd_pred + other.prd_pred
        metric.prd_gold = self.prd_gold + other.prd_gold
        metric.prd_cm = self.prd_cm + other.prd_cm
        metric.tp = self.tp + other.tp
        metric.pred = self.pred + other.pred
        metric.gold = self.gold + other.gold
        return metric

    @classmethod
    def span2prop(cls, spans, length):
        prds, args = ['-'] * length, {}
        for prd, start, end, role in spans:
            prds[prd-1] = str(prd)
            if prd not in args:
                args[prd] = ['*'] * length
                args[prd][prd-1] = '(V*)'
            args[prd][start-1] = f'({role}*'
            args[prd][end-1] += ')'
        args = [args[key] for key in sorted(args)]
        return '\n'.join(['\t'.join(i) for i in zip(*[prds, *args])])

    @property
    def score(self):
        return self.f

    @property
    def prd_f(self):
        return 2 * self.prd_tp / (self.prd_pred + self.prd_gold + self.eps)

    @property
    def cm(self):
        return self.prd_cm / (self.prd_gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)
