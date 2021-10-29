# -*- coding: utf-8 -*-

import argparse

from tqdm import tqdm


def process(prop, fid, file):
    with open(fid) as f:
        ids = {line[25:].strip() for line in f}
    with open(prop) as fprop, open(file, 'w') as f:
        written = False
        for line in tqdm(fprop):
            cols = line.split()
            if len(cols) < 1:
                if not written:
                    f.write(line)
                    written = True
            elif cols[0] in ids:
                if cols[7] == '-':
                    cols[6] = '-'
                f.write('\t'.join([cols[3], cols[6], *cols[11:-1]]) + '\n')
                written = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Filter annotations by id file.'
    )
    parser.add_argument('--prop', help='path to the prop file')
    parser.add_argument('--fid', help='path to the id file')
    parser.add_argument('--file', help='path to the saved file')
    args = parser.parse_args()
    process(args.prop, args.fid, args.file)
