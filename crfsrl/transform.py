# -*- coding: utf-8 -*-

from supar.utils.transform import CoNLL


class CoNLL(CoNLL):
    r"""
    The CoNLL object holds ten fields required for CoNLL-X data format :cite:`buchholz-marsi-2006-conll`.
    Each field can be bound to one or more :class:`~supar.utils.field.Field` objects. For example,
    ``FORM`` can contain both :class:`~supar.utils.field.Field` and :class:`~supar.utils.field.SubwordField`
    to produce tensors for words and subwords.

    Attributes:
        ID:
            Token counter, starting at 1.
        FORM:
            Words in the sentence.
        LEMMA:
            Lemmas or stems (depending on the particular treebank) of words, or underscores if not available.
        CPOS:
            Coarse-grained part-of-speech tags, where the tagset depends on the treebank.
        POS:
            Fine-grained part-of-speech tags, where the tagset depends on the treebank.
        FEATS:
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or underscores if not available.
        HEAD:
            Heads of the tokens, which are either values of ID or zeros.
        DEPREL:
            Dependency relations to the HEAD.
        PHEAD:
            Projective heads of tokens, which are either values of ID or zeros, or underscores if not available.
        PDEPREL:
            Dependency relations to the PHEAD, or underscores if not available.
    """

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self,
                 ID=None, FORM=None, LEMMA=None, CPOS=None, POS=None,
                 FEATS=None, HEAD=None, DEPREL=None, PHEAD=None, PDEPREL=None):
        super().__init__()

        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.CPOS = CPOS
        self.POS = POS
        self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.PHEAD = PHEAD
        self.PDEPREL = PDEPREL

    @property
    def src(self):
        return self.FORM, self.LEMMA, self.CPOS, self.POS, self.FEATS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL, self.PHEAD, self.PDEPREL

    @classmethod
    def get_arcs(cls, sequence, placeholder='_'):
        return [-1 if i == placeholder else int(i) for i in sequence]

    @classmethod
    def get_sibs(cls, sequence, placeholder='_'):
        sibs = [[0] * (len(sequence) + 1) for _ in range(len(sequence) + 1)]
        heads = [0] + [-1 if i == placeholder else int(i) for i in sequence]

        for i, hi in enumerate(heads[1:], 1):
            for j, hj in enumerate(heads[i+1:], i + 1):
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i][hi] = j
                    else:
                        sibs[j][hj] = i
                    break
        return sibs[1:]

    @classmethod
    def get_edges(cls, sequence):
        edges = [[0]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edges[i][int(pair.split(':')[0])] = 1
        return edges

    @classmethod
    def get_labels(cls, sequence):
        labels = [[None]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edge, label = pair.split(':')
                    labels[i][int(edge)] = label
        return labels

    @classmethod
    def get_srl_edges(cls, sequence):
        edges = [[[False]*(len(sequence)+1) for _ in range(len(sequence)+1)] for _ in range(len(sequence)+1)]
        spans = [['O']*len(sequence) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence):
            if s != '_':
                for pair in s.split('|'):
                    head, label = pair.split(':')
                    if label != '[prd]':
                        spans[int(head)][i] = label
                    else:
                        spans[i + 1][i] = label
        for i, label in enumerate(sequence):
            edges[i+1][i+1][0] = '[prd]' in label

        def factorize(tags):
            spans = []
            for i, tag in enumerate(tags, 1):
                if tag.startswith('B'):
                    spans.append([i, i+1, tag[2:]])
                elif tag.startswith('O') and (len(spans) == 0 or spans[-1][-1] != 'O'):
                    spans.append([i, i+1, 'O'])
                elif tag.startswith('['):
                    spans.append([i, i+1, tag])
                else:
                    spans[-1][1] += 1
            return spans
        for prd, arg_labels in enumerate(spans[1:], 1):
            for *span, label in factorize(arg_labels):
                if span[0] != prd:
                    if label != 'O':
                        edges[prd][span[0] if span[0] < prd else span[1]-1][prd] = True
                    else:
                        for i in range(*span):
                            edges[prd][i][prd] = True
                for i in range(*span):
                    if i != prd:
                        for j in range(*span):
                            if i != j:
                                edges[prd][i][j] = True
        return edges

    @classmethod
    def get_srl_roles(cls, sequence):
        labels = [['O']*(len(sequence)+1) for _ in range(len(sequence)+1)]
        spans = [['O']*len(sequence) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence):
            if s != '_':
                for pair in s.split('|'):
                    head, label = pair.split(':')
                    if label != '[prd]':
                        spans[int(head)][i] = label
                    else:
                        spans[i + 1][i] = label

        def factorize(tags):
            spans = []
            for i, tag in enumerate(tags, 1):
                if tag.startswith('B'):
                    spans.append([i, i+1, tag[2:]])
                elif tag.startswith('O') and (len(spans) == 0 or spans[-1][-1] != 'O'):
                    spans.append([i, i+1, 'O'])
                elif tag.startswith('['):
                    spans.append([i, i+1, tag])
                else:
                    spans[-1][1] += 1
            return spans
        for prd, arg_labels in enumerate(spans[1:], 1):
            if '[prd]' not in sequence[prd-1]:
                continue
            labels[prd][0] = '[prd]'
            for *span, label in factorize(arg_labels):
                if not label.startswith('['):
                    for i in range(*span):
                        labels[prd][i] = label
            labels[prd][prd] = '[prd]'
        return labels

    @classmethod
    def get_srl_spans(cls, sequence):
        labels = []
        spans = [['O']*len(sequence) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence):
            if s != '_':
                for pair in s.split('|'):
                    head, label = pair.split(':')
                    if label != '[prd]':
                        spans[int(head)][i] = label
                    else:
                        spans[i + 1][i] = label

        def factorize(tags):
            spans = []
            for i, tag in enumerate(tags, 1):
                if tag.startswith('B'):
                    spans.append([i, i+1, tag[2:]])
                elif tag.startswith('O') and (len(spans) == 0 or spans[-1][-1] != 'O'):
                    spans.append([i, i+1, 'O'])
                elif tag.startswith('['):
                    spans.append([i, i+1, tag])
                else:
                    spans[-1][1] += 1
            return spans
        for prd, arg_labels in enumerate(spans[1:], 1):
            if '[prd]' not in sequence[prd-1]:
                continue
            for i, j, label in factorize(arg_labels):
                if i != prd and not label.startswith('['):
                    labels.append((prd, i, j-1, label))
        return labels

    @classmethod
    def build_relations(cls, chart):
        sequence = ['_'] * len(chart)
        for i, row in enumerate(chart):
            pairs = [(j, label) for j, label in enumerate(row) if label is not None]
            if len(pairs) > 0:
                sequence[i] = '|'.join(f"{head}:{label}" for head, label in pairs)
        return sequence

    @classmethod
    def build_srl_roles(cls, spans, length):
        labels = [''] * length
        for span in spans:
            prd, head, start, end, label = span
            if label == 'O':
                continue
            if '[prd]' not in labels[prd-1]:
                labels[prd-1] = '|'.join((labels[prd-1], '0:[prd]'))
            labels[start-1] = '|'.join((labels[start-1], f'{prd}:B-{label}'))
            for i in range(start, end):
                labels[i] = '|'.join((labels[i], f'{prd}:I-{label}'))
        labels = [('_' if not label else label).lstrip('|') for label in labels]
        return labels

    @classmethod
    def factorize(cls, tags):
        spans = []
        for i, tag in enumerate(tags, 1):
            if tag.startswith('B'):
                spans.append([i, i+1, tag[2:]])
            elif tag.startswith('I'):
                if len(spans) > 0 and spans[-1][-1] == tag[2:]:
                    spans[-1][1] += 1
                else:
                    spans.append([i, i+1, tag[2:]])
            elif len(spans) == 0 or spans[-1][-1] != tag:
                spans.append([i, i+1, tag])
            else:
                spans[-1][1] += 1
        return spans
