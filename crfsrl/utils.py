# -*- coding: utf-8 -*-


def union_find(sequence, roots=None):
    sequence = [0] + sequence
    sets, connected = {root: {root} for root in (roots or [0])}, False
    while not connected:
        connected = True
        for i, head in enumerate(sequence):
            if i in sets:
                pass
            elif head in sets:
                sets[head].add(i)
            elif head != 0:
                connected = False
                sequence[i] = sequence[head]
    return [sorted(i) for i in sets.values()]
