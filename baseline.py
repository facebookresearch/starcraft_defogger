# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torchcraft import replayer
from data import DataLoader
from utils import setup, normalize_by_first_dim, hash_map, get_info, get_normalize_fn
from itertools import chain
from threading import Lock
import os
import _ext
import argparse
import logging
import warnings
warnings.filterwarnings("ignore", "", DeprecationWarning)

parser = argparse.ArgumentParser(
    description='The best defogger for a misty winter day\n\n'
                'Recall that our dumped replays have a innate skip-frame of 3. '
                'This baseline simply returns the average feature vector over '
                '(race, time, map) for the given input.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--method', default="stat", type=str,
                    help="Which method to use for a baseline:"
                         "\n| stat - return average over (race, time, map, start_location) from training set"
                         "\n| prev - use previous frame")

args = setup(parser)
if args.save:
    logging.error("Save is bugged right now, the dictionary we generate is "
                  "200 GB and it will crash when you try to save it. "
                  "TODO: make the array an hdf5 instead if we want to reuse.")

stride = args.kernel_size if args.stride == 0 else args.stride
featurizer = _ext.CoarseConvFeaturizerUnitTypes(
    args.kernel_size, args.kernel_size, stride, stride)


def get_targ_pred(pred, targ):
    if args.predict == "full_info":
        return targ[:-1], targ[1:]
    elif args.predict == "defog":
        return pred, targ
    else:
        raise RuntimeError("--predict has to be one of {full_info, defog}")


all_data = {}
global_lock = Lock()
locks = {}


def get_featurize_fn(ret, method):
    def featurize(fn):
        rep = replayer.load(fn)
        map = rep.getMap()
        map_size = map['walkability'].shape

        batch = []
        for i in range(0, len(rep), args.skip_frames):
            frames = [rep.getFrame(k) for k in
                      range(i, min(i + args.combine_frames, len(rep)))]
            batch.append(frames)

        if method == 'prev':
            return [
                featurizer.featurize(batch, map_size, perspective=0),
                featurizer.featurize(batch, map_size, perspective=1),
                featurizer.featurize(batch, map_size, perspective=0, full=True),
                featurizer.featurize(batch, map_size, perspective=1, full=True),
            ]

        elif method == 'stat':
            features = featurizer.featurize(batch, map_size, full=True)
            map_hash = hash_map(map)
            '''
            Two players with the same info is considered to be the same 'game'
            for this baseline: All ZvT mu that start at the same location
            should be hashed to a single info. Should be irrespective of p1
            or p2, so we sort.
            '''
            info_hash = tuple(sorted(get_info(rep)))

            if ret:
                return (map_hash, info_hash, features)

            key = (map_hash, info_hash)
            with global_lock:
                if key in all_data:
                    data_tup = all_data[key]
                else:
                    all_data[key] = [np.zeros(features.shape), 0]
                    locks[key] = Lock()
            with locks[key]:
                data_tup = all_data[key]
                acc = data_tup[0]
                if acc.shape[0] < features.shape[0]:
                    acc = np.pad(
                        acc,
                        ((0, features.shape[0] - acc.shape[0]), (0,0), (0,0), (0,0)),
                        mode='constant',
                        constant_values=0)
                view = acc[tuple(slice(0, int(axis)) for axis in features.shape)]
                view += features
                view += features
                data_tup[0] = acc
                data_tup[1] += 1
                all_data[key] = data_tup

            return True
        else:
            raise RuntimeError("No such way to calculate a baseline. "
                               "Check your --method")
    return featurize


if args.method == "stat":
    if not args.load or not os.path.exists(args.load + ".npy"):
        dl = DataLoader(args, get_featurize_fn(False, args.method))
        dl.train()
        logging.info("No previous model, recreating statistics")
        for data in dl:
            pass

        if args.save:
            logging.info("Saving to {}".format(args.save + 'npy'))
            np.save(args.save, all_data)
    else:
        logging.info("Loading model from {}".format(args.load + '.npy'))
        all_data = np.load(args.load + ".npy", all_data)
        all_data = all_data[()]  # unpack the dictionary

normalize = get_normalize_fn(args)

logging.info("Running baseline on test set with method {}".format(args.method))
dl = DataLoader(args, get_featurize_fn(True, args.method))
dl.test()
results = []
for data in dl:
    if args.method == "stat":
        map, info, target = data
        # must predict one frame ahead, so we discard first frame.
        # We also normalize per frame, by dividing by # of units per frame
        # NOTE: This actuall predicts the distribution rather than the count.

        target = target[1:]
        target = normalize(target)

        # Default predicts all zeros, we can probably do slightly better by
        # marginalizing across races or something? TODO
        data_tup = all_data.get((map, info), [np.zeros(target.shape), 1])
        prediction = data_tup[0] / data_tup[1]
        prediction = prediction[1:]
        prediction = normalize(prediction)

        # full_info and defog are the same here...

        # We repeated predict the last frame if the game is longer
        # than any games in the dataset
        if prediction.shape[0] < target.shape[0]:
            prediction = np.pad(
                prediction,
                ((0, target.shape[0] - prediction.shape[0]), (0, 0), (0, 0), (0, 0)),
                mode='edge')

        view = prediction[tuple(slice(0, int(axis)) for axis in target.shape)]
        err = ((target - view)**2).mean(axis=(1, 2, 3))
        results += err.tolist()

    elif args.method == "prev":
        p0, p1, t0, t1 = [normalize(x) for x in data]
        t0 = t0[1:]
        t1 = t1[1:]
        p0 = p0[:-1]
        p1 = p1[:-1]

        p0, t0 = get_targ_pred(p0, t0)
        p1, t1 = get_targ_pred(p1, t1)

        err = ((t0 - p0)**2).mean(axis=(1, 2, 3))
        results += err.tolist()
        err = ((t1 - p1)**2).mean(axis=(1, 2, 3))
        results += err.tolist()


print("MSE = {}".format(sum(results) / len(results)))
