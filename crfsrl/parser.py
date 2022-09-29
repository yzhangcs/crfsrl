# -*- coding: utf-8 -*-

import os
from typing import Iterable, Union

import torch
from supar.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, MIN, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.fn import pad
from supar.utils.logging import get_logger
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch

from .metric import SpanSRLMetric
from .model import CRF2oSemanticRoleLabelingModel, CRFSemanticRoleLabelingModel
from .transform import CoNLL

logger = get_logger(__name__)


class CRFSemanticRoleLabelingParser(Parser):
    r"""
    The implementation of Semantic Role Labeling Parser using span-constrained CRF.
    """

    NAME = 'crf-semantic-role-labeling'
    MODEL = CRFSemanticRoleLabelingModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.EDGE, self.ROLE, self.SPAN = self.transform.PHEAD

    def train(
        self,
        train: Union[str, Iterable],
        dev: Union[str, Iterable],
        test: Union[str, Iterable],
        epochs: int = 1000,
        patience: int = 100,
        batch_size: int = 5000,
        update_steps: int = 1,
        buckets: int = 32,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().train(**Config().update(locals()))

    def evaluate(
        self,
        data: Union[str, Iterable],
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().evaluate(**Config().update(locals()))

    def predict(
        self,
        data: Union[str, Iterable],
        pred: str = None,
        lang: str = None,
        prob: bool = False,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, edges, roles, spans = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        s_edge, s_role = self.model(words, feats)
        loss = self.model.loss(s_edge, s_role, edges, roles, mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SpanSRLMetric:
        words, *feats, edges, roles, spans = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        s_edge, s_role = self.model(words, feats)
        loss = self.model.loss(s_edge, s_role, edges, roles, mask)
        role_preds = self.model.decode(s_edge, s_role, mask)
        return SpanSRLMetric(
            loss,
            [[(i[0], *i[2:-1], self.ROLE.vocab[i[-1]]) for i in s if i[-1] != self.ROLE.unk_index] for s in role_preds],
            [[i for i in s if i[-1] != 'O'] for s in spans]
        )

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, *feats = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        lens = mask.sum(-1)
        s_edge, s_role = self.model(words, feats)
        if self.args.prd:
            prd_mask = pad([word_mask.new_tensor([0]+['[prd]' in i for i in s.values[8]]) for s in batch.sentences])
            s_edge[:, 0].masked_fill_(~prd_mask, MIN)
            s_role[..., 0, 0].masked_fill_(prd_mask, MIN)
            s_role[..., 0, self.args.prd_index].masked_fill_(~prd_mask, MIN)
        role_preds = [[(*i[:-1], self.ROLE.vocab[i[-1]]) for i in s] for s in self.model.decode(s_edge, s_role, mask)]
        batch.roles = [CoNLL.build_srl_roles(pred, length) for pred, length in zip(role_preds, lens.tolist())]
        if self.args.prob:
            scores = zip(*(s.cpu().unbind() for s in (s_edge, s_role)))
            batch.probs = [(s[0][:i+1, :i+1], s[1][:i+1, :i+1]) for i, s in zip(lens.tolist(), scores)]
        return batch

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
        TAG, CHAR, LEMMA, ELMO, BERT = None, None, None, None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                t = TransformerTokenizer(args.bert)
                BERT = SubwordField('bert', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
                BERT.vocab = t.vocab
        EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_srl_edges)
        ROLE = ChartField('roles', unk='O', fn=CoNLL.get_srl_roles)
        SPAN = RawField('spans', fn=CoNLL.get_srl_spans)
        transform = CoNLL(FORM=(WORD, CHAR, ELMO, BERT), LEMMA=LEMMA, POS=TAG, PHEAD=(EDGE, ROLE, SPAN))

        train = Dataset(transform, args.train)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        ROLE.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_roles': len(ROLE.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'prd_index': ROLE.vocab['[prd]'],
            'nul_index': ROLE.vocab['O']
        })
        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser


class CRF2oSemanticRoleLabelingParser(CRFSemanticRoleLabelingParser):
    r"""
    The implementation of Semantic Role Labeling Parser using second-order span-constrained CRF.
    """

    NAME = 'crf2o-semantic-role-labeling'
    MODEL = CRF2oSemanticRoleLabelingModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.EDGE, self.ROLE, self.SPAN = self.transform.PHEAD

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, edges, roles, spans = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        s_edge, s_sib, s_role = self.model(words, feats)
        loss = self.model.loss(s_edge, s_sib, s_role, edges, roles, mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SpanSRLMetric:
        words, *feats, edges, roles, spans = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        s_edge, s_sib, s_role = self.model(words, feats)
        loss = self.model.loss(s_edge, s_sib, s_role, edges, roles, mask)
        role_preds = self.model.decode(s_edge, s_sib, s_role, mask)
        return SpanSRLMetric(
            loss,
            [[(i[0], *i[2:-1], self.ROLE.vocab[i[-1]]) for i in s if i[-1] != self.ROLE.unk_index] for s in role_preds],
            [[i for i in s if i[-1] != 'O'] for s in spans]
        )

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, *feats = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        lens = mask.sum(-1)
        s_edge, s_sib, s_role = self.model(words, feats)
        if self.args.prd:
            prd_mask = pad([word_mask.new_tensor([0]+['[prd]' in i for i in s.values[8]]) for s in batch.sentences])
            s_edge[:, 0].masked_fill_(~prd_mask, MIN)
            s_role[..., 0, 0].masked_fill_(prd_mask, MIN)
            s_role[..., 0, self.args.prd_index].masked_fill_(~prd_mask, MIN)
        role_preds = [[(*i[:-1], self.ROLE.vocab[i[-1]]) for i in s]
                      for s in self.model.decode(s_edge, s_sib, s_role, mask)]
        batch.roles = [CoNLL.build_srl_roles(pred, length) for pred, length in zip(role_preds, lens.tolist())]
        if self.args.prob:
            scores = zip(*(s.cpu().unbind() for s in (s_edge, s_sib, s_role)))
            batch.probs = [(s[0][:i+1, :i+1], s[1][:i+1, :i+1, :i+1], s[2][:i+1, :i+1])
                           for i, s in zip(lens.tolist(), scores)]
        return batch

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
        TAG, CHAR, LEMMA, ELMO, BERT = None, None, None, None, None
        if args.encoder == 'bert':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField('words',
                                pad=t.pad_token,
                                unk=t.unk_token,
                                bos=t.bos_token or t.cls_token,
                                fix_len=args.fix_len,
                                tokenize=t.tokenize,
                                fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField('bert',
                                    pad=t.pad_token,
                                    unk=t.unk_token,
                                    bos=t.bos_token or t.cls_token,
                                    fix_len=args.fix_len,
                                    tokenize=t.tokenize,
                                    fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
                BERT.vocab = t.get_vocab()
        EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_srl_edges)
        ROLE = ChartField('roles', unk='O', fn=CoNLL.get_srl_roles)
        SPAN = RawField('spans', fn=CoNLL.get_srl_spans)
        transform = CoNLL(FORM=(WORD, CHAR, ELMO, BERT), LEMMA=LEMMA, POS=TAG, PHEAD=(EDGE, ROLE, SPAN))

        train = Dataset(transform, args.train)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        ROLE.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_roles': len(ROLE.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'prd_index': ROLE.vocab['[prd]'],
            'nul_index': ROLE.vocab['O']
        })
        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser
