# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchcraft import replayer
import torch as th
import numpy as np
import random
import io
import gzip

from utils import get_info, make_start_loc, get_normalize_fn, convert_minutes_to_repframes


def get_featurize_fn(args, pers):
    normalize = get_normalize_fn(args)
    featurizer = args.featurizer

    # TODO:
    # * check that it has enemy units that we see in input
    # * add resources for us (and maybe resources for the enemy in target)
    def featurize_reduced(fn):
        with np.load(fn) as data:
            info = data['info']
            race = [int(info[0]), int(info[3])]
            map_size = (info[6], info[7])
            # XXX and another copy...
            rdata = data['reduced'].tobytes()

            mfdata = io.BytesIO(data['map'])
        with gzip.GzipFile(fileobj=mfdata, mode='r') as f:
            map_features = np.load(f)

        vis = args.use_true_visibility
        feats = featurizer.featurize_reduced_all(
            rdata, args.skip_frames, args.combine_frames, map_size, pers,
            visibility=vis, ground_height=map_features[2])
        input = feats[0]
        targets = feats[1]
        if vis:
            visibility = feats[2]
        else:
            visibility = input.sum(-1, keepdims=True) > 0
        assert input.shape[0] > 1

        featurized = (
            # map
            th.from_numpy(map_features[np.newaxis, :]).type(th.FloatTensor),
            # race
            th.LongTensor([race if pers == 0 else list(reversed(race))]),
            # input
            th.from_numpy(normalize(input.transpose(0, 3, 1, 2))[:-1]).type(th.FloatTensor),
            # targets
            th.from_numpy(normalize(targets.transpose(0, 3, 1, 2))[1:]).type(th.FloatTensor),
            # game name
            fn,
            # visibility
            visibility.transpose(0, 3, 1, 2),
        )
        return featurized

    def featurize(fn):
        raise RuntimeError("This doesn't work anymore, add in game name, visibility")
        rep = replayer.load(fn)
        map = rep.getMap()
        map_size = map['walkability'].shape
        race = [x[0] for x in get_info(rep)]
        sl = make_start_loc(map['start_locations'], map_size)

        batch = []
        fromt = args.from_time if args.until_time != 0 else 0
        until = args.from_time if args.until_time != 0 else len(rep)
        for i in range(fromt, min(len(rep), until), args.skip_frames):
            frames = [rep.getFrame(k) for k in
                      range(i, min(i + args.combine_frames, len(rep)))]
            batch.append(frames)

        featurized = (
            # map
            th.from_numpy(
                np.stack([
                    map['walkability'],
                    map['buildability'],
                    map['ground_height'],
                    sl,
                ], axis=0)[np.newaxis, :]).type(th.FloatTensor),
            # race
            th.LongTensor([race if pers == 0 else list(reversed(race))]),
            # input
            th.from_numpy(
                normalize(featurizer.featurize(
                    batch, map_size, perspective=pers
                ).transpose(0, 3, 1, 2))[:-1]).type(th.FloatTensor),
            # targets
            th.from_numpy(
                normalize(featurizer.featurize(
                    batch, map_size, perspective=pers, full=True
                ).transpose(0, 3, 1, 2))[1:]).type(th.FloatTensor),
        )
        assert featurized[2].size(0) > 1

        return featurized

    if args.reduced:
        return featurize_reduced
    return featurize


def get_featurize_fn_mock(args, pers):
    def featurize(fn):
        w = random.randint(2, 8) * 128
        h = random.randint(2, 8) * 128
        t = random.randint(5000, 10000) // args.skip_frames
        sw = (w - args.kernel_size) // args.stride + 1
        sh = (h - args.kernel_size) // args.stride + 1
        return (
            th.zeros(1, 4, h, w).type(th.FloatTensor),
            th.LongTensor([0, 1]),
            th.zeros(t, 236, sh, sw).type(th.FloatTensor),
            th.zeros(t, 236, sh, sw).type(th.FloatTensor),
        )
    return featurize



def add_common_arguments(parser):
    parser.add_argument('--lr', default=0.01, type=float,
            help="learning rate")
    parser.add_argument('--lr_decay', default=False, action='store_true',
            help="whether to decay the learning rate when loss stalls")
    parser.add_argument('--weight_decay', default=0, type=float,
            help="What weight decay to use, _as a proportion of the learning rate_")
    parser.add_argument('--momentum', default=0, type=float,
            help="What momentum to use, only applicable for SGD and RMSPROP")
    parser.add_argument('--bptt', default=0, type=int,
            help="BPTT length, 0 means full sequence")
    parser.add_argument('--conv_bptt', default=0, type=int,
            help="Convolutional BPTT length, 0 means full sequence")
    parser.add_argument('--clip', default=-1.0, type=float,
            help="gradient norm clipping param")
    parser.add_argument('--plot', nargs='?', const='http://localhost', type=str,
            help="--plot plots to visdom on localhost, but you can specifiy --plot server, port is fixed to be 8097")
    parser.add_argument('--valid_every', default=10000, type=int,
            help="how often to valid and plot the result of inference")
    parser.add_argument('--loss', default='MSE', type=str,
            help="REGRESSION loss to use | {'MSE', 'SmoothL1', 'SoftMargin'}")
    parser.add_argument('--loss_averaging', default=False, action='store_true',
            help="whether to size_average the loss")
    parser.add_argument('--regression_lambda', default=1, type=float,
            help="lambda classifying the existence of units")
    parser.add_argument('--unit_loss_lambda', default=0, type=float,
            help="lambda classifying the existence of units")
    parser.add_argument('--bldg_loss_lambda', default=0, type=float,
            help="lambda classifying the existence of buildings")
    parser.add_argument('--opbt_loss_lambda', default=0, type=float,
            help="lambda classifying the existence of opponent build tree")
    parser.add_argument('--optim', default='SGD', type=str,
            help="optimizer to use can be any found in torch.optim")
    parser.add_argument('--n_input_timesteps', default=1, type=int,
            help="number of time steps as input [1: input=s_t], e.g. 2=(s_{t-1}, s_{t})")

    return parser
