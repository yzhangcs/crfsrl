# -*- coding: utf-8 -*-

import argparse

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    maps = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return [lemmatizer.lemmatize(w, maps[p[0].upper()]) if p[0].upper() in maps else w for w, p in nltk.pos_tag(words)]


def build_roles(spans, length):
    rels = [''] * length
    for span in spans:
        prd, start, end, label = span
        if label == 'O':
            continue
        if label == '[prd]':
            rels[prd-1] = '|'.join((rels[prd-1], '0:[prd]'))
            continue
        rels[start-1] = '|'.join((rels[start-1], f'{prd}:B-{label}'))
        for i in range(start, end):
            rels[i] = '|'.join((rels[i], f'{prd}:I-{label}'))
    rels = [('_' if not label else label).lstrip('|') for label in rels]
    return rels


def prop2conllu(lines):
    words = [line.split()[0] for line in lines]
    lemmas, pred_lemmas = [line.split()[1] for line in lines], lemmatize(words)
    lemmas = [i if i != '-' else pred for i, pred in zip(lemmas, pred_lemmas)]
    spans = []

    if len(lines[0].split()) >= 2:
        prds, *args = list(zip(*[line.split()[1:] for line in lines]))
        prds = [i for i, p in enumerate(prds, 1) if p != '-']
        for i, p in enumerate(prds):
            spans.append((p, 1, len(words), '[prd]'))
            starts, rels = zip(*[(j, a.split('*')[0].split('(')[1]) for j, a in enumerate(args[i], 1) if a.startswith('(')])
            ends = [j for j, a in enumerate(args[i], 1) if a.endswith(')')]
            for s, r, e in zip(starts, rels, ends):
                if r == 'V':
                    continue
                spans.append((p, s, e, r))
    roles = build_roles(spans, len(words))
    return ['\t'.join([str(i), word, lemma, '_', '_', '_', '_', '_', role, '_'])
            for i, (word, lemma, role) in enumerate(zip(words, lemmas, roles), 1)]


def process(prop, file):
    with open(prop) as f:
        lines = [line.strip() for line in f]
    i, start, sentences = 0, 0, []
    for line in tqdm(lines):
        if not line:
            sentences.append(prop2conllu(lines[start:i]))
            start = i + 1
        i += 1
    with open(file, 'w') as f:
        for s in sentences:
            f.write('\n'.join(s) + '\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert the file of prop format to conllu format.'
    )
    parser.add_argument('--prop', help='path to the prop file')
    parser.add_argument('--file', help='path to the converted conllu file')
    args = parser.parse_args()
    process(args.prop, args.file)
