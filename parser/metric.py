# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile

import torch
from supar.utils.metric import Metric

from .utils import download


class SpanSRLMetric(Metric):

    URL = 'http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz'
    DATA_PATH = os.path.expanduser(os.path.join(torch.hub.DEFAULT_CACHE_DIR, 'supar', 'datasets', 'srl'))

    def __init__(self, eps=1e-12):
        super().__init__()

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.prd_tp = 0
        self.prd_pred = 0
        self.prd_gold = 0
        self.eps = eps

        self.script = os.path.join(self.DATA_PATH, 'srlconll-1.1/bin/srl-eval.pl')
        if not os.path.exists(self.script):
            download(self.URL, os.path.join(self.DATA_PATH, 'srlconll-1.1.tgz'))

    def __call__(self, preds, golds):
        lens = [max(max([*i, *j], key=lambda x: max(x[:3]))[:3]) if i or j else 1 for i, j in zip(preds, golds)]
        fpred, fgold = os.path.join(tempfile.mkdtemp(), 'pred'), os.path.join(tempfile.mkdtemp(), 'gold')
        with open(fpred, 'w') as f:
            f.write('\n\n'.join([self.span2prop(spans, lens[i]) for i, spans in enumerate(preds)]))
        with open(fgold, 'w') as f:
            f.write('\n\n'.join([self.span2prop(spans, lens[i]) for i, spans in enumerate(golds)]))
        os.environ['PERL5LIB'] = os.path.join(self.DATA_PATH, 'srlconll-1.1/lib') + os.pathsep + os.environ.get('PERL5LIB', '')
        p_out = subprocess.check_output(['perl', f'{self.script}', f'{fpred}', f'{fgold}'], stderr=subprocess.STDOUT).decode()
        r_out = subprocess.check_output(['perl', f'{self.script}', f'{fgold}', f'{fpred}'], stderr=subprocess.STDOUT).decode()
        p_out = [i for i in p_out.split('\n') if 'Overall' in i][0].split()
        r_out = [i for i in r_out.split('\n') if 'Overall' in i][0].split()
        os.remove(fpred)
        os.remove(fgold)

        self.tp += int(p_out[1])
        self.pred += int(p_out[3]) + int(p_out[1])
        self.gold += int(r_out[3]) + int(p_out[1])
        for pred, gold in zip(preds, golds):
            prd_pred, prd_gold = {span[0] for span in pred}, {span[0] for span in gold}
            self.prd_tp += len(prd_pred & prd_gold)
            self.prd_pred += len(prd_pred)
            self.prd_gold += len(prd_gold)
        return self

    def __repr__(self):
        return f"PRD: {self.prd_p:6.2%} {self.prd_r:6.2%} {self.prd_f:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

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
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)

    @property
    def prd_p(self):
        return self.prd_tp / (self.prd_pred + self.eps)

    @property
    def prd_r(self):
        return self.prd_tp / (self.prd_gold + self.eps)

    @property
    def prd_f(self):
        return 2 * self.prd_tp / (self.prd_pred + self.prd_gold + self.eps)
