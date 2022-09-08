# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, MIN, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.fn import pad
from supar.utils.logging import get_logger, progress_bar
from supar.utils.parallel import parallel
from supar.utils.tokenizer import TransformerTokenizer

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

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, cache=False,
              clip=5.0, epochs=5000, patience=100, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (str or Iterable):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, cache=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
                The data for evaluation. Both a filename and a list of instances are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, cache=False, prob=False,
                verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
                The data for prediction.
                - a filename. If ends with `.txt`, the parser will seek to make predictions line by line from plain texts.
                - a list of instances.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @parallel()
    def _train(self, loader):
        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            words, *feats, edges, roles, spans = batch.compose(self.transform)
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_role = self.model(words, feats)
                loss = self.model.loss(s_edge, s_role, edges, roles, mask)
                loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = SpanSRLMetric()

        for batch in loader:
            words, *feats, edges, roles, spans = batch.compose(self.transform)
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_role = self.model(words, feats)
                loss = self.model.loss(s_edge, s_role, edges, roles, mask)
            role_preds = self.model.decode(s_edge, s_role, mask)
            metric += SpanSRLMetric(
                loss,
                [[(i[0], *i[2:-1], self.ROLE.vocab[i[-1]]) for i in s if i[-1] != self.ROLE.unk_index] for s in role_preds],
                [[i for i in s if i[-1] != 'O'] for s in spans]
            )
        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, *feats = batch.compose(self.transform)
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0
            lens = mask.sum(-1)
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_role = self.model(words, feats)
            prd_mask = pad([word_mask.new_tensor([0]+['[prd]' in i for i in s.values[8]]) for s in batch.sentences])
            if self.args.prd:
                s_edge[:, 0].masked_fill_(~prd_mask, MIN)
                s_role[..., 0, 0].masked_fill_(prd_mask, MIN)
                s_role[..., 0, self.args.prd_index].masked_fill_(~prd_mask, MIN)
            role_preds = [[(*i[:-1], self.ROLE.vocab[i[-1]]) for i in s] for s in self.model.decode(s_edge, s_role, mask)]
            batch.roles = [CoNLL.build_srl_roles(pred, length) for pred, length in zip(role_preds, lens.tolist())]
            if self.args.prob:
                scores = zip(*(s.cpu().unbind() for s in (s_edge, s_role)))
                batch.probs = [(s[0][:i+1, :i+1], s[1][:i+1, :i+1]) for i, s in zip(lens.tolist(), scores)]
            yield from batch.sentences

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

    def train(self, train, dev, test, buckets=32, workers=0, batch_size=5000, update_steps=1, cache=False,
              clip=5.0, epochs=5000, patience=100, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (str or Iterable):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, workers=0, batch_size=5000, cache=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
                The data for evaluation. Both a filename and a list of instances are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, workers=0, batch_size=5000, cache=False, prob=False,
                verbose=True, **kwargs):
        r"""
        Args:
            data (str or Iterable):
                The data for prediction.
                - a filename. If ends with `.txt`, the parser will seek to make predictions line by line from plain texts.
                - a list of instances.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        return super().predict(**Config().update(locals()))

    @parallel()
    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)

        for i, batch in enumerate(bar, 1):
            words, *feats, edges, roles, spans = batch.compose(self.transform)
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_sib, s_role = self.model(words, feats)
                loss = self.model.loss(s_edge, s_sib, s_role, edges, roles, mask)
                loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

    @parallel(training=False)
    def _evaluate(self, loader):
        metric = SpanSRLMetric()

        for batch in loader:
            words, *feats, edges, roles, spans = batch.compose(self.transform)
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_sib, s_role = self.model(words, feats)
                loss = self.model.loss(s_edge, s_sib, s_role, edges, roles, mask)
            role_preds = self.model.decode(s_edge, s_sib, s_role, mask)
            metric += SpanSRLMetric(
                loss,
                [[(i[0], *i[2:-1], self.ROLE.vocab[i[-1]]) for i in s if i[-1] != self.ROLE.unk_index] for s in role_preds],
                [[i for i in s if i[-1] != 'O'] for s in spans]
            )

        return metric

    @parallel(training=False, op=None)
    def _predict(self, loader):
        for batch in progress_bar(loader):
            words, *feats = batch.compose(self.transform)
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask[:, 0] = 0
            lens = mask.sum(-1)
            with torch.autocast(self.device, enabled=self.args.amp):
                s_edge, s_sib, s_role = self.model(words, feats)
            from supar.utils.fn import pad
            prd_mask = pad([word_mask.new_tensor([0]+['[prd]' in i for i in s.values[8]]) for s in batch.sentences])
            if self.args.prd:
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
            yield from batch.sentences

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
