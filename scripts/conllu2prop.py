# -*- coding: utf-8 -*-

import argparse

from tqdm import tqdm


def seq2span(sequence):
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
    roles = []
    for prd, arg_roles in enumerate(spans[1:], 1):
        if '[prd]' not in sequence[prd-1]:
            continue
        for i, j, role in factorize(arg_roles):
            if i != prd and not role.startswith('[') and role != 'O':
                roles.append((prd, i, j-1, role))
    return roles


def span2prop(lines):
    cols = list(zip(*[line.split() for line in lines]))
    words, lemmas, roles = cols[1], cols[2], seq2span(cols[8])
    lemma_length = max([len(i) for i in lemmas]) + 1
    if len(roles) > 0:
        role_length = max([len(i[-1]) for i in roles]) + 1
    prds, args = ['-' + ' ' * (lemma_length-1)] * len(words), {}
    for prd, start, end, role in roles:
        prds[prd-1] = lemmas[prd-1] + ' ' * (lemma_length-len(lemmas[prd-1]))
        if prd not in args:
            args[prd] = [' ' * (role_length) + '* '] * len(words)
            args[prd][prd-1] = ' ' * (role_length-2) + '(V*)'
        args[prd][start-1] = ' ' * (role_length-len(role)-1) + f'({role}* '
        args[prd][end-1] = args[prd][end-1][:-1] + ')'
    args = [args[key] for key in sorted(args)]
    return '\n'.join([' '.join(i) for i in zip(*[prds, *args])])


def process(conllu, file):
    with open(conllu) as f:
        lines = [line.strip() for line in f]
    i, start = 0, 0
    with open(file, 'w') as f:
        for line in tqdm(lines):
            if not line:
                f.write(span2prop(lines[start:i])+'\n\n')
                start = i + 1
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Recover the conllu file to prop file.'
    )
    parser.add_argument('--conllu', help='path to the conllu file')
    parser.add_argument('--file', help='path to the converted prop file')
    args = parser.parse_args()
    process(args.conllu, args.file)
