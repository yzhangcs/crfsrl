# -*- coding: utf-8 -*-

import os
import sys
import urllib
import zipfile

import torch
import tarfile


def download(url, dst=None, reload=False):
    path = dst
    if path is None:
        path = os.path.join(os.path.expanduser('~/.cache/supar'), os.path.basename(urllib.parse.urlparse(url).path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if reload:
        os.remove(path) if os.path.exists(path) else None
    if not os.path.exists(path):
        sys.stderr.write(f"Downloading: {url} to {path}\n")
        try:
            torch.hub.download_url_to_file(url, path, progress=True)
        except urllib.error.URLError:
            raise RuntimeError(f"File {url} unavailable. Please try other sources.")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as f:
            members = f.infolist()
            path = os.path.join(os.path.dirname(path), members[0].filename)
            if len(members) != 1:
                raise RuntimeError('Only one file (not dir) is allowed in the zipfile.')
            if reload or not os.path.exists(path):
                f.extractall(os.path.dirname(path))
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tar:
            for i in tar:
                tar.extract(i, os.path.dirname(path))
    return path


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
