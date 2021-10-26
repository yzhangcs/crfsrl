# -*- coding: utf-8 -*-

import argparse

import stanza
from tqdm import tqdm

pipeline = stanza.Pipeline(lang='en', processors='tokenize,lemma', verbose=False, tokenize_no_ssplit=True)


def build_roles(spans, length):
    rels = [''] * length
    for span in spans:
        prd, start, end, label = span
        if label == 'O':
            continue
        if '[prd]' not in rels[prd-1]:
            rels[prd-1] = '|'.join((rels[prd-1], '0:[prd]'))
        rels[start-1] = '|'.join((rels[start-1], f'{prd}:B-{label}'))
        for i in range(start, end):
            rels[i] = '|'.join((rels[i], f'{prd}:I-{label}'))
    rels = [('_' if not label else label).lstrip('|') for label in rels]
    return rels


def prop2conllu(lines):
    words = [line.split()[0] for line in lines]
    lemmas = [f'{word.lemma}' for word in pipeline(' '.join(words)).sentences[0].words]
    spans = []
    if len(lines[0].split()[1:]) > 1:
        prds, *args = list(zip(*[line.split()[1:] for line in lines]))
        prds = [i for i, p in enumerate(prds, 1) if p != '-']
        for i, p in enumerate(prds):
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
