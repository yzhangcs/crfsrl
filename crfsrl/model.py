# -*- coding: utf-8 -*-

import torch
from supar.model import Model
from supar.modules import Biaffine, Triaffine
from supar.utils import Config
from supar.utils.common import MIN

from .struct import SRLCRF, SRL2oCRF
from .utils import union_find


class CRFSemanticRoleLabelingModel(Model):
    r"""
    The implementation of Semantic Role Labeling Parser using span-constrained CRF.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_roles (int):
            The number of roles in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        freeze (bool):
            If ``True``, freezes BERT parameters, required if using BERT features. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_encoder_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_edge_mlp (int):
            Unary factor MLP size. Default: 500.
        mlp_dropout (float):
            The dropout ratio of unary edge factor MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_roles,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['char', 'lemma'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_encoder_hidden=400,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_edge_mlp=500,
                 n_role_mlp=100,
                 mlp_dropout=.33,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.edge_attn = Biaffine(n_in=self.args.n_encoder_hidden,
                                  n_proj=n_edge_mlp,
                                  dropout=mlp_dropout,
                                  bias_x=True,
                                  bias_y=False)
        self.role_attn = Biaffine(n_in=self.args.n_encoder_hidden,
                                  n_out=n_roles,
                                  n_proj=n_role_mlp,
                                  dropout=mlp_dropout,
                                  bias_x=True,
                                  bias_y=True)

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first and last are scores of all possible edges (``[batch_size, seq_len, seq_len]``)
                and possible roles on each edge (``[batch_size, seq_len, seq_len, n_roles]``).
        """

        x = self.encode(words, feats)
        s_edge = self.edge_attn(x, x)
        s_role = self.role_attn(x, x).permute(0, 2, 3, 1)
        role_mask = s_role.new_ones(self.args.n_roles).gt(0)
        role_mask[[self.args.nul_index, self.args.prd_index]] = 0
        s_role[..., self.args.nul_index] = 0
        s_role[..., 0, role_mask] = MIN
        s_role[..., 1:, self.args.prd_index] = MIN
        return s_edge, s_role

    def loss(self, s_edge, s_role, edges, roles, mask):
        prd_mask = edges[..., 0].diagonal(0, 1, 2)
        if self.args.prd:
            s_edge[:, 0].masked_fill_(~prd_mask, MIN)
            s_role[..., 0, 0].masked_fill_(prd_mask, MIN)
            s_role[..., 0, self.args.prd_index].masked_fill_(~prd_mask, MIN)
        p_role = s_role.log_softmax(-1)
        loss = -p_role[:, 1:, 0, 0]
        if prd_mask.any():
            prds = torch.where(prd_mask)[1]
            s_mask = (prd_mask*prds.new_tensor(range(mask.shape[0])).unsqueeze(-1))[prd_mask]
            edges, roles = edges[prd_mask], roles[prd_mask]
            s_edge, p_role = s_edge[s_mask], p_role[s_mask]
            s_edge[..., 0].masked_fill_(prds.unsqueeze(-1).ne(prds.new_tensor(range(mask.shape[1]))), MIN)
            s_role = torch.zeros_like(p_role[..., 0])
            s_role[range(len(prds)), prds, 0] = p_role[range(len(prds)), prds, 0, self.args.prd_index]
            s_role[range(len(prds)), :, prds] = p_role[range(len(prds)), :, prds].gather(-1, roles.unsqueeze(-1)).squeeze(-1)
            edge_dist, role_dist = SRLCRF(s_edge, mask[s_mask]), SRLCRF(s_edge + s_role, mask[s_mask])
            loss = loss.masked_scatter(prd_mask[:, 1:], edge_dist.log_partition - role_dist.score(edges))
        return loss[mask[:, 1:]].sum() / mask.sum()

    def decode(self, s_edge, s_role, mask):
        lens = mask.sum(-1)
        batch_size, seq_len = mask.shape
        edge_preds = lens.new_zeros(batch_size, seq_len, seq_len)
        roles = s_role.argmax(-1)
        prd_mask = roles[..., 0].eq(self.args.prd_index).index_fill(1, lens.new_tensor(0), 0)
        if prd_mask.any():
            prds = torch.where(prd_mask)[1]
            s_mask = (prd_mask*prds.new_tensor(range(batch_size)).unsqueeze(-1))[prd_mask]
            s_edge, s_role = s_edge[s_mask], s_role[s_mask]
            s_role, p_role = torch.zeros_like(s_role[..., 0]), s_role.log_softmax(-1)
            s_role[range(len(prds)), prds, 0] = p_role[range(len(prds)), prds, 0, self.args.prd_index]
            p_role = p_role[range(len(prds)), :, prds].gather(-1, roles.transpose(1, 2)[prd_mask].unsqueeze(-1)).squeeze(-1)
            s_role[range(len(prds)), :, prds] = p_role
            s_edge[..., 0].masked_fill_(prds.unsqueeze(-1).ne(prds.new_tensor(range(mask.shape[1]))), MIN)
            edge_preds[prd_mask] = SRLCRF(s_edge + s_role, mask[s_mask]).argmax
        role_preds = roles.unsqueeze(1).repeat(1, seq_len, 1, 1).gather(-1, edge_preds.unsqueeze(-1)).squeeze(-1)
        arc_heads = edge_preds.eq(edge_preds.new_tensor(range(seq_len)).unsqueeze(-1))
        preds = [[] for _ in range(batch_size)]
        for i, length in zip(range(batch_size), (lens+1).tolist()):
            for prd, (arc_pred, rel_pred, heads) in enumerate(zip(edge_preds[i], role_preds[i], arc_heads[i])):
                heads = torch.where(heads[:length])[0].tolist()
                if not 0 < prd < length or not prd_mask[i, prd] or len(heads) == 0:
                    continue
                spans = union_find(arc_pred[1:length].tolist(), heads)
                roles = rel_pred[arc_pred.eq(prd)].tolist()
                preds[i].extend([(prd, head, span[0], span[-1], role) for head, span, role in zip(heads, spans, roles)])
        return preds


class CRF2oSemanticRoleLabelingModel(Model):
    r"""
    The implementation of Semantic Role Labeling Parser using second-order span-constrained CRF.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_roles (int):
            The number of roles in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        freeze (bool):
            If ``True``, freezes BERT parameters, required if using BERT features. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of LSTM hidden states. Default: 600.
        n_encoder_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_edge_mlp (int):
            Unary factor MLP size. Default: 500.
        mlp_dropout (float):
            The dropout ratio of unary edge factor MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_roles,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['char', 'lemma'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_encoder_hidden=400,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_edge_mlp=500,
                 n_sib_mlp=100,
                 n_role_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.edge_attn = Biaffine(n_in=self.args.n_encoder_hidden,
                                  n_proj=n_edge_mlp,
                                  dropout=mlp_dropout,
                                  bias_x=True,
                                  bias_y=False)
        self.sib_attn = Triaffine(n_in=self.args.n_encoder_hidden,
                                  n_proj=n_sib_mlp,
                                  dropout=mlp_dropout,
                                  bias_x=True,
                                  bias_y=True)
        self.role_attn = Biaffine(n_in=self.args.n_encoder_hidden,
                                  n_out=n_roles,
                                  n_proj=n_role_mlp,
                                  dropout=mlp_dropout,
                                  bias_x=True,
                                  bias_y=True)

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.
        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible edges (``[batch_size, seq_len, seq_len]``),
                second-order siblings (``[batch_size, seq_len, seq_len, seq_len]``),
                and possible roles on each edge (``[batch_size, seq_len, seq_len, n_roles]``).
        """

        x = self.encode(words, feats)
        s_edge = self.edge_attn(x, x)
        s_sib = self.sib_attn(x, x, x).permute(0, 3, 1, 2)
        s_role = self.role_attn(x, x).permute(0, 2, 3, 1)
        role_mask = s_role.new_ones(self.args.n_roles).gt(0)
        role_mask[[0, self.args.prd_index]] = 0
        s_role[..., self.args.nul_index] = 0
        s_role[..., 0, role_mask] = MIN
        s_role[..., 1:, self.args.prd_index] = MIN
        return s_edge, s_sib, s_role

    def loss(self, s_edge, s_sib, s_role, edges, roles, mask):
        prd_mask = edges[..., 0].diagonal(0, 1, 2)
        if self.args.prd:
            s_edge[:, 0].masked_fill_(~prd_mask, MIN)
            s_role[..., 0, 0].masked_fill_(prd_mask, MIN)
            s_role[..., 0, self.args.prd_index].masked_fill_(~prd_mask, MIN)
        p_role = s_role.log_softmax(-1)
        loss = -p_role[:, 1:, 0, 0]
        if prd_mask.any():
            prds = torch.where(prd_mask)[1]
            s_mask = (prd_mask*prds.new_tensor(range(mask.shape[0])).unsqueeze(-1))[prd_mask]
            edges, roles = edges[prd_mask], roles[prd_mask]
            s_edge, s_sib, p_role = s_edge[s_mask], s_sib[s_mask], p_role[s_mask]
            s_edge[..., 0].masked_fill_(prds.unsqueeze(-1).ne(prds.new_tensor(range(mask.shape[1]))), MIN)
            s_role = torch.zeros_like(p_role[..., 0])
            s_role[range(len(prds)), prds, 0] = p_role[range(len(prds)), prds, 0, self.args.prd_index]
            s_role[range(len(prds)), :, prds] = p_role[range(len(prds)), :, prds].gather(-1, roles.unsqueeze(-1)).squeeze(-1)
            edge_dist, role_dist = SRL2oCRF((s_edge, s_sib), mask[s_mask]), SRL2oCRF((s_edge + s_role, s_sib), mask[s_mask])
            loss = loss.masked_scatter(prd_mask[:, 1:], edge_dist.log_partition - role_dist.score((edges, roles)))
        return loss[mask[:, 1:]].sum() / mask.sum()

    def decode(self, s_edge, s_sib, s_role, mask):
        lens = mask.sum(-1)
        batch_size, seq_len = mask.shape
        edge_preds = lens.new_zeros(batch_size, seq_len, seq_len)
        p_role, roles = s_role.log_softmax(-1).max(-1)
        prd_mask = roles[..., 0].eq(self.args.prd_index) & mask
        if prd_mask.any():
            prds = torch.where(prd_mask)[1]
            s_mask = (prd_mask*prds.new_tensor(range(batch_size)).unsqueeze(-1))[prd_mask]
            s_edge, s_sib, p_role = s_edge[s_mask], s_sib[s_mask], p_role[s_mask]
            s_edge[..., 0].masked_fill_(prds.unsqueeze(-1).ne(prds.new_tensor(range(mask.shape[1]))), MIN)
            s_role = torch.zeros_like(p_role)
            s_role[range(len(prds)), prds, 0] = p_role[range(len(prds)), prds, 0]
            s_role[range(len(prds)), :, prds] = p_role[range(len(prds)), :, prds]
            edge_preds[prd_mask] = SRL2oCRF((s_edge + s_role, s_sib), mask[s_mask]).argmax
        role_preds = roles.unsqueeze(1).repeat(1, seq_len, 1, 1).gather(-1, edge_preds.unsqueeze(-1)).squeeze(-1)
        arc_heads = edge_preds.eq(edge_preds.new_tensor(range(seq_len)).unsqueeze(-1))
        preds = [[] for _ in range(batch_size)]
        for i, length in zip(range(batch_size), (lens+1).tolist()):
            for prd, (arc_pred, rel_pred, heads) in enumerate(zip(edge_preds[i], role_preds[i], arc_heads[i])):
                heads = torch.where(heads[:length])[0].tolist()
                if not 0 < prd < length or not prd_mask[i, prd] or len(heads) == 0:
                    continue
                spans = union_find(arc_pred[1:length].tolist(), heads)
                roles = rel_pred[arc_pred.eq(prd)].tolist()
                preds[i].extend([(prd, head, span[0], span[-1], role) for head, span, role in zip(heads, spans, roles)])
        return preds
