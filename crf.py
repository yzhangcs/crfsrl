# -*- coding: utf-8 -*-

import argparse

from crfsrl import CRFSemanticRoleLabelingParser
from supar.cmds.cmd import init
from supar.utils.common import CACHE


def main():
    parser = argparse.ArgumentParser(description='Create first-order CRF Dependency Parser.')
    parser.set_defaults(Parser=CRFSemanticRoleLabelingParser)
    parser.add_argument('--prd', action='store_true', help='whether to use gold predicates')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'lemma', 'elmo', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--finetune', action='store_true', help='whether to finetune PLM models')
    subparser.add_argument('--encoder', choices=['lstm', 'transformer', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default=f'{CACHE}/data/srl/conll05/train.conllu', help='path to train file')
    subparser.add_argument('--dev', default=f'{CACHE}/data/srl/conll05/dev.conllu', help='path to dev file')
    subparser.add_argument('--test', default=f'{CACHE}/data/srl/conll05/test.conllu', help='path to test file')
    subparser.add_argument('--embed', default='glove-6b-100', help='path to pretrained embeddings')
    subparser.add_argument('--bert', default='bert-large-cased', help='which BERT model to use')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default=f'{CACHE}/data/srl/conll05/test.conllu', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default=f'{CACHE}/data/srl/conll05/test.conllu', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    init(parser)


if __name__ == "__main__":
    main()
