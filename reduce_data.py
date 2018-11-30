# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torchcraft import replayer
from data import DataLoader
from utils import setup, hash_map, make_start_loc, get_info
from itertools import chain
from threading import Lock
from os import path
import gzip
import os
import io
import _ext
import argparse
import logging

parser = argparse.ArgumentParser(
    description='The best defogger for a misty winter day\n\n'
                'Recall that our dumped replays have a innate skip-frame of 3. '
                'This reduces the replay data to an npz file. The name for '
                'the dump file is {args.save}/#/*.npz '
                '\nThe format of the dir is: '
                '\n\t#/replay.tcr.npz/reduced - binary data for featurizer'
                '\n\t#/replay.tcr.npz/map - 4xHxW of the maps, walk, build, height, start_loc'
                '\n\t#/replay.tcr.npz/features.attr - 10 tuple of (p0race, p0start_x, p0start_y, p1race, .., .., map_hash, .., .., ..)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--list', default = '', help="Use given list")
parser.add_argument('--skip-existing', action='store_true', help='Skip existing data')

args = setup(parser)

stride = args.kernel_size if args.stride == 0 else args.stride
featurizer = _ext.CoarseConvFeaturizerUnitTypes(
    args.kernel_size, args.kernel_size, stride, stride)

if args.save:
    base = args.save
    try:
        os.mkdir(base)
        files = [os.mkdir("{}/{}".format(base, i)) for i in range(20)]
    except:
        pass
else:
    raise RuntimeError("How do you expect me to dump when you don't tell me"
                       " where to save to???")

def featurize(fn):
    savep = "{}/{}".format(path.basename(path.dirname(fn)), path.basename(fn))
    savep = path.join(base, savep) + '.npz'
    os.makedirs(path.dirname(savep), exist_ok=True)
    if args.skip_existing and path.exists(savep):
        logging.info("Found existing {}, skipping".format(savep))
        return True

    if not os.path.exists(fn):
        raise RuntimeError("No such replay")
    rep = replayer.load(fn)
    map = rep.getMap()
    map_size = map['walkability'].shape
    sl = make_start_loc(map['start_locations'], map_size)

    batch = []
    for i in range(0, len(rep)):
        batch.append(rep.getFrame(i))
    reduced = featurizer.reduce(batch)
    map_features = np.stack([
        map['walkability'],
        map['buildability'],
        map['ground_height'],
        sl,
    ], axis=0).astype(dtype='uint8')

    info_hash = chain(*get_info(rep))  # 6 ints, p0race, p0slx, p0sly, p1...
    map_hash = hash_map(map)  # 4 ints for the map hash

    info = np.asarray(list(chain(info_hash, map_hash)), dtype=np.int32)

    mfdata = io.BytesIO()
    with gzip.GzipFile(fileobj=mfdata, mode='w') as f:
        np.save(f, map_features)
    to_save = {
        'reduced': reduced,
        'map': mfdata.getvalue(),
        'info': info,
    }
    # TODO: Both reduced data and overall data may be compressed
    np.savez(savep, **to_save)

    return True


dl = DataLoader(args, featurize)
if args.list:
    dl.list(args.list)
else:
    dl.all()
logging.info("Begin feature reduction...")
for d in dl:
    pass

logging.info("Done!")
