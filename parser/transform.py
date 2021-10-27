# -*- coding: utf-8 -*-

import os

from supar.utils.logging import progress_bar
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import CoNLLSentence, Transform


class CoNLL(Transform):
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
    def toconll(cls, tokens):
        r"""
        Converts a list of tokens to a string in CoNLL-X format.
        Missing fields are filled with underscores.

        Args:
            tokens (list[str] or list[tuple]):
                This can be either a list of words, word/pos pairs or word/lemma/pos triples.

        Returns:
            A string in CoNLL-X format.

        Examples:
            >>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
            1       She     _       _       _       _       _       _       _       _
            2       enjoys  _       _       _       _       _       _       _       _
            3       playing _       _       _       _       _       _       _       _
            4       tennis  _       _       _       _       _       _       _       _
            5       .       _       _       _       _       _       _       _       _

            >>> print(CoNLL.toconll([('She',     'she',    'PRP'),
                                     ('enjoys',  'enjoy',  'VBZ'),
                                     ('playing', 'play',   'VBG'),
                                     ('tennis',  'tennis', 'NN'),
                                     ('.',       '_',      '.')]))
            1       She     she     PRP     _       _       _       _       _       _
            2       enjoys  enjoy   VBZ     _       _       _       _       _       _
            3       playing play    VBG     _       _       _       _       _       _
            4       tennis  tennis  NN      _       _       _       _       _       _
            5       .       _       .       _       _       _       _       _       _

        """

        if isinstance(tokens[0], str):
            s = '\n'.join([f"{i}\t{word}\t" + '\t'.join(['_']*8)
                           for i, word in enumerate(tokens, 1)])
        elif len(tokens[0]) == 2:
            s = '\n'.join([f"{i}\t{word}\t_\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, tag) in enumerate(tokens, 1)])
        elif len(tokens[0]) == 3:
            s = '\n'.join([f"{i}\t{word}\t{lemma}\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, lemma, tag) in enumerate(tokens, 1)])
        else:
            raise RuntimeError(f"Invalid sequence {tokens}. Only list of str or list of word/pos/lemma tuples are support.")
        return s + '\n'

    @classmethod
    def isprojective(cls, sequence):
        r"""
        Checks if a dependency tree is projective.
        This also works for partial annotation.

        Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
        which are hard to detect in the scenario of partial annotation.

        Args:
            sequence (list[int]):
                A list of head indices.

        Returns:
            ``True`` if the tree is projective, ``False`` otherwise.

        Examples:
            >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
            False
            >>> CoNLL.isprojective([3, -1, 2])
            False
        """

        pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
        for i, (hi, di) in enumerate(pairs):
            for hj, dj in pairs[i+1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                    return False
        return True

    @classmethod
    def istree(cls, sequence, proj=False, multiroot=False):
        r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (list[int]):
                A list of head indices.
            proj (bool):
                If ``True``, requires the tree to be projective. Default: ``False``.
            multiroot (bool):
                If ``False``, requires the tree to contain only a single root. Default: ``True``.

        Returns:
            ``True`` if the arcs form an valid tree, ``False`` otherwise.

        Examples:
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

        from supar.utils.alg import tarjan
        if proj and not cls.isprojective(sequence):
            return False
        n_roots = sum(head == 0 for head in sequence)
        if n_roots == 0:
            return False
        if not multiroot and n_roots > 1:
            return False
        if any(i == head for i, head in enumerate(sequence, 1)):
            return False
        return next(tarjan(sequence), None) is None

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

    def load(self, data, lang=None, proj=False, max_len=None, **kwargs):
        r"""
        Loads the data in CoNLL-X format.
        Also supports for loading data from CoNLL-U file with comments and non-integer IDs.

        Args:
            data (list[list] or str):
                A list of instances or a filename.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            proj (bool):
                If ``True``, discards all non-projective sentences. Default: ``False``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`CoNLLSentence` instances.
        """

        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        else:
            if lang is not None:
                tokenizer = Tokenizer(lang)
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = '\n'.join([self.toconll(i) for i in data]).split('\n')

        i, start, sentences = 0, 0, []
        for line in progress_bar(lines):
            if not line:
                sentences.append(CoNLLSentence(self, lines[start:i]))
                start = i + 1
            i += 1
        if proj:
            sentences = [i for i in sentences if self.isprojective(list(map(int, i.arcs)))]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences
