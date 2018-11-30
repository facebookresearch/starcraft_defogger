# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import _ext
import torchcraft
import torch as th
from torch import nn
import os

import numpy as np

import argparse, sys, random
from itertools import product
from collections import OrderedDict

from utils import setup, zero_init, div_init
from compare_models import ModelCompare
import conv_lstm_models as M
from conv_lstm_models import get_GatedConvolution
from conv_lstm_utils import add_common_arguments

parser = argparse.ArgumentParser(
    description='The best defogger for a misty winter day\n\n'
                'Recall that our dumped replays have a innate skip-frame of 3.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--run', default=None, type=str,
                    help="Which experiment to run. If not specified, we search through experiments")

parser.add_argument('--exp', default='007', type=str,
                    help="display info about experiment set")
parser.add_argument('--ls', default=False, action='store_true',
                    help="display info about experiment set")
parser.add_argument('-f', '--filter', default=[], action='append',nargs=2, metavar=('field','value'),
                    help='Can do this multiple times for an "and". --filter conv conv2d will find all runs that use conv2d. Use CMD to search through the cmdline args')
parser.add_argument('--top', default=0, type=int,
                    help="Restrict to top n experiments")
parser.add_argument('--random', default=False, action='store_true',
                    help="Pick some random experiments from this experiment set")
parser.add_argument('--print', default=False, action='store_true',
                    help="print out the arguments")
parser.add_argument('--validate', default=None, type=str,
                    help="validates experiment ids, argument is the file to use")
parser.add_argument('--no_checkpoint', default=False, action='store_true',
                    help="Don't checkpoint")
parser.add_argument('--just_valid', default=False, action='store_true',
                    help="Only do a validation run")
parser.add_argument('--just_test', default=False, action='store_true',
                    help="Only do a test run")
parser.add_argument('--dump_npz', default=False, action='store_true',
                    help="Only do a test run")
parser.add_argument('--class_prob_thresh', default=0.5, type=float,
                    help="Existence is measured by when p[building] is greater than this flag")
parser.add_argument('--use_true_visibility', default=False, action='store_true',
                    help="Use Jonas's visibility map, which is strictly more visible than reality. Default is strictly less visible")
parser.add_argument('--n_unit_thresh', default=0.1, type=float,
                    help="If model predicts > n_unit_thresh, round it to the next whole number")
parser.add_argument('--regr_slope_scalar', default=1.0, type=float,
                    help="If model predict K units, return K*regr_slope_scalar units")

args = parser.parse_args()
program_args = args

experiments = {}
exp_set = {}  # Holds arguments for filtering

savestr = "--load" if args.just_valid or args.just_test else "--io"

experiments['test'] = (
    (((savestr + " /tmp/test ") if not args.no_checkpoint else "") +
    ("-e 1 " if not args.just_test else "-e 0 ") +
    "--lr 0.00001 "
    "--optim Adam "
    "--predict 'only_defog' "
    "--predict_delta "
    "-k 64 "
    "-s 32 "
    "--divide_by 10 "
    "--opbt_loss_lambda 1 "
    "--bldg_loss_lambda 1 "
    "--unit_loss_lambda 1 "
    "--regression_lambda 1 "
    " -i /datasets01/starcraft_replays-reduced/060817/ "
    "--debug 2 "
    "--until_time 15 "
    "--clip 10 "
    "--valid_every 150 "
    "--small "
    ).split(),
    [
        lambda a: M.multilvl_lstm(a, conv=nn.Conv2d, nonlin=nn.ELU, hid_dim=256, residual=False, bypass_encoder=True, map_embsize=8),
    ]
)

def add_default_cmdline(cmdline):
    return cmdline + (
        " --reduced --data_threads 10 -g 0 "
        + (" --use_true_visibility" if args.use_true_visibility else "")
        + (" --class_prob_thresh {}".format(args.class_prob_thresh))
        + (" --n_unit_thresh {}".format(args.n_unit_thresh))
        + (" --regr_slope_scalar {}".format(args.regr_slope_scalar))
        + (" --just_valid " if args.just_valid else "")
    ).split()


def merge(*args):
    ret = {}
    for x in args: ret.update(x)
    return ret

def make_k_s_fs_args(x):
    if x is None:
        return x
    k, s, fs, rest = x
    return "-k {} -s {} --skip_frames {fs} --combine_frames {fs} {rest}".format(k, s, fs=fs, rest=rest)

def fon(s, x):
    if x is None:
        return x
    return s.format(x)

def add_experiments_007():
    '''
    WARNING BEFORE YOU CHANGE THIS:
        1. The serializer depends on you None-ing out things you don't want to try anymore
        2. Always add to the end of every list.
        3. New argument sets go at the end
    '''
    def make_kw_depth(kw, depth):
        return {
            'enc_convsize': kw,
            'dec_convsize': kw,
            'enc_depth': depth,
            'dec_depth': depth,
        }

    def mkcmdline(strings):
        return [None if s is None else (s, {}) for s in strings]

    def mkkwargs(kwargs):
        return [None if k is None else ("", k) for k in kwargs]

    default_cmdline = (
        ("-e 200 " if not args.just_test else "-e 0 ") +
        " --divide_by 10"
        " --clip 10"
        " --valid_every 15000"
        " --debug 2"
    )

    default_ops = {'map_embsize': 8}

    model_sets = [
        ("str-kw3", [(M.striding, merge(
                make_kw_depth(3, x),
                {'model_name': 's_depth{}'.format(x)},
                default_ops,
        )) for x in [4, 9]]),
        ("str-kw5", [(M.striding, merge(
                make_kw_depth(5, x),
                {'model_name': 's_depth{}'.format(x)},
                default_ops,
            )) for x in [4, 9]]),
        ("ml", [(M.multilvl_lstm, merge(
                {'model_name': 'ml_depth{}'.format(x), 'n_lvls': x, 'midconv_depth': x},
                default_ops,
            )) for x in [2, 3]]),
        ("2d1d-kw3", [(M.conv_only, merge(
                make_kw_depth(3, x),
                {'model_name': 'c_depth{}'.format(x)},
                default_ops,
            )) for x in [4, 9]]),
        ("2d1d-kw5", [(M.conv_only, merge(
                make_kw_depth(5, x),
                {'model_name': 'c_depth{}'.format(x)},
                default_ops,
            )) for x in [4, 9]]),
    ]

    # just cmdline
    k_s_fs = mkcmdline([make_k_s_fs_args(x) for x in [
        (128, 64, 120, "--predict 'defog' --predict_delta"),
        None, # (128, 64, 40, "--predict 'defog' --predict_delta"),
        (64, 32, 40, "--predict 'defog' --predict_delta"),
        None, # (64, 32, 80, "--predict 'defog' --predict_delta"),
        (64, 32, 120, "--predict 'defog' --predict_delta"),
        (32, 32, 40, "--predict 'defog' --predict_delta"),
        None, # (64, 64, 40, "--predict 'defog' --predict_delta"),
        None, # (64, 64, 120, "--predict 'defog' --predict_delta"),
        (32, 32, 120, "--predict 'defog' --predict_delta"),
        (32, 32, 40, "--predict 'only_defog' --predict_delta"),
        (32, 32, 240, "--predict 'defog' --predict_delta"),
        (128, 64, 120, "--predict 'defog'"),
        (32, 32, 120, "--predict 'defog'"),
        (32, 32, 40, "--predict 'only_defog'"),
        (64, 32, 240, "--predict 'defog' --predict_delta"),
        (64, 32, 240, "--predict 'defog'"),
        (64, 32, 40, "--predict 'only_defog' --predict_delta"),
        (64, 32, 40, "--predict 'only_defog'"),
        (32, 32, 8, "--bptt 40 --predict 'only_defog' --predict_delta"),
        (32, 32, 8, "--bptt 40 --predict 'defog' --predict_delta"),
    ]])

    losses = mkcmdline([
        None, # "--loss MSE",
        "--loss SmoothL1",
        None, #"--loss SmoothL1 --regression_lambda 0 --unit_loss_lambda 0.5 --bldg_loss_lambda 0.5 ",  # Just classification
        None, # "--loss SmoothL1 --regression_lambda .03 --unit_loss_lambda 0.5 --bldg_loss_lambda 0.5 ",
        None, # "--loss SmoothL1 --regression_lambda .1 --unit_loss_lambda 0.5 --bldg_loss_lambda 0.5 ",
        None, #"--loss SmoothL1 --regression_lambda .3 --unit_loss_lambda 0.5 --bldg_loss_lambda 0.5 ",
        "--loss SmoothL1 --regression_lambda 1 --opbt_loss_lambda 0.01 ",
        None, #"--loss SmoothL1 --regression_lambda 1 --opbt_loss_lambda 0.001 ",
        "--loss SmoothL1 --regression_lambda 1 --opbt_loss_lambda 0.1 ",
    ])

    lrs = mkcmdline([
        None, #'--lr 1e-5',
        '--lr 1e-4',
        None, #'--lr 1e-3',
        None, #'--lr 1e-6',
        None, # '--lr 1e-2',
        "",  # defer to optimizers
    ])

    optimizers = mkcmdline([
        "--optim SGD --momentum 0.99",
        "--optim Adam ",
        None, #"--optim SGD --momentum 0.90",
        "--optim SGD ",
        "--optim Adam --lr_decay",
        "--optim SGD --lr_decay",
        "--optim SGD --lr 3e-2",
    ])

    # just kwargs
    module_types = mkkwargs([
        {'conv':nn.Conv2d, 'nonlin':nn.ReLU, 'residual':False},
        None, #{'conv':nn.Conv2d, 'nonlin':nn.ELU, 'residual':False},
        {'conv':get_GatedConvolution(nn.Conv2d), 'nonlin':None, 'residual':False},
        None, #{'conv':nn.Conv2d, 'nonlin':nn.ReLU, 'residual':True},
        None, #{'conv':nn.Conv2d, 'nonlin':nn.SELU, 'residual':False},
        None, #{'conv':nn.Conv2d, 'nonlin':nn.ELU, 'residual':True},
        {'conv':nn.Conv2d, 'nonlin':nn.ReLU, 'residual':True},
        {'conv':nn.Conv2d, 'nonlin':nn.ReLU, 'residual':False, 'hid_dim': 128, 'inp_embsize': 128, 'enc_embsize': 128, 'bypass_encoder': True},
        {'conv':get_GatedConvolution(nn.Conv2d), 'nonlin':None, 'residual':False, 'hid_dim': 128, 'inp_embsize': 128, 'enc_embsize': 128, 'bypass_encoder': True},
    ])

    # both
    reg = [
        ("", {}),
        None, #("--weight_decay 0.01", {}),  # This is proportion of LR
        ("", {'lstm_dropout': 0.1}),
        None, # ("", {'lstm_dropout': 0.3}),
        ("--bptt 20", {'lstm_dropout': 0.1}),
    ]

    with_z = mkkwargs([
        {'with_z': False},
        {'with_z': True, 'z_after_lstm': True, 'zbwd_single': False},
        {'with_z': True, 'z_after_lstm': True, 'zbwd_single': True},
        {'with_z': True, 'z_after_lstm': False, 'zbwd_single': False},
        {'with_z': True, 'z_after_lstm': False, 'zbwd_single': True},
    ])

    z_optparam = mkkwargs([
        {},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e-1, 'z_lr':0.1},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e0, 'z_lr':0.1},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e1, 'z_lr':0.1},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e2, 'z_lr':0.1},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e3, 'z_lr':0.1},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e4, 'z_lr':0.1},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e-1, 'z_lr':0.01},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e0, 'z_lr':0.01},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e1, 'z_lr':0.01},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e2, 'z_lr':0.01},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e3, 'z_lr':0.01},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e4, 'z_lr':0.01},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e-1, 'z_lr':0.01},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e0, 'z_lr':0.01},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e1, 'z_lr':0.01},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e2, 'z_lr':0.01},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e3, 'z_lr':0.01},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e4, 'z_lr':0.01},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e-1, 'z_lr':0.001},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e0, 'z_lr':0.001},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e1, 'z_lr':0.001},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e2, 'z_lr':0.001},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e3, 'z_lr':0.001},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e4, 'z_lr':0.001},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e-1, 'z_lr':0.1, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e0, 'z_lr':0.1, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e1, 'z_lr':0.1, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e2, 'z_lr':0.1, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e3, 'z_lr':0.1, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e4, 'z_lr':0.1, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e-1, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e0, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e1, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e2, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e3, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.SGD, 'z_lambda': 1e4, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e-1, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e0, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e1, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e2, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e3, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e4, 'z_lr':0.01, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e-1, 'z_lr':0.001, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e0, 'z_lr':0.001, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e1, 'z_lr':0.001, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e2, 'z_lr':0.001, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e3, 'z_lr':0.001, 'zfwd_zbwd_ratio': 0.5},
        {'z_opt': th.optim.Adam, 'z_lambda': 1e4, 'z_lr':0.001, 'zfwd_zbwd_ratio': 0.5},
    ])

    clip = mkcmdline([
        " --clip 10 ",
        " --clip 500 ",
    ])

    datasrc = mkcmdline([
        "",
        " -i /path/to/aiide17/data ",
        " --finetune ", # finetune specialization code generated elsewhere
    ])

    timeclip = mkcmdline([
        " --until_time 11 --from_time 3 ",
        " --finetune "
    ])

    def inc(lst, maxes, ind=None):
        if ind is None:
            ind = len(lst) - 1
        while ind >= 0:
            lst[ind] += 1
            if lst[ind] == maxes[ind]:
                lst[ind] = 0
                ind -= 1
            else:
                break
        else:
            return False
        return True


    def serialize(*args):
        maximum = 12
        if len(args) == maximum:
            raise RuntimeError("Serialization will break, please increase the maximum")

        stuff = [list(enumerate(x)) for x in args]
        maxes = [len(x) for x in stuff]
        inds = [0 for i in stuff]
        if program_args.run is not None:
            codestring = program_args.run[-maximum*2:]
            codes = [int(codestring[2*x:2*x+2]) for x in range(len(codestring)//2)]
            inds = codes[:len(inds)]
        while True:
            elements = [stuff[i][x] for i, x in enumerate(inds)]
            if not inc(inds, maxes):
                break

            nums = [x[0] for x in elements]
            bad = False
            for i, x in enumerate(elements):
                if x[1] is None or x[1][0] is None:
                    inc(inds, maxes, i)
                    bad = True
                    break
            if bad:
                continue

            cmdlineargs = " ".join(x[1][0] for x in elements) + default_cmdline
            kwargargs = merge(*(x[1][1] for x in elements))

            bad = False
            for i, x in enumerate(kwargargs):
                if x is None:
                    inc(inds, maxes, i)
                    bad = True
                    break
            if bad:
                continue
            padding = "00" * (maximum - len(nums))
            id = "".join(str(x).zfill(2) for x in nums) + padding

            yield (nums, id, cmdlineargs, kwargargs)

    generator = serialize(
        k_s_fs,
        losses,
        lrs,
        optimizers,
        module_types,
        reg,
        with_z,
        z_optparam,
        clip,
        datasrc,
        timeclip,
    )

    def filter_stupid(name, nums):
        # conv models can't have lstm dropout
        if '2d1d' in name and nums[5] in [2,3]:
            return True
        # need to specify lr in optimizer if not in lrs
        if nums[2] == 5 and nums[3] not in [6]:
            return True
        if nums[6] == 0 and nums[7] != 0:
            return True
        return False


    searchable = OrderedDict()
    exp_set['007'] = searchable
    for nums, id, orig_cmd, kwargs in generator:
        for model_name, models in model_sets:
            cmd = orig_cmd
            if filter_stupid(model_name, nums):
                continue
            name = "007_{}_{}".format(model_name, id)
            pathstr = " {} ./defogger/{} "
            if not args.no_checkpoint:
                extras = pathstr.format(savestr, name)
                cmd = cmd + extras
            if "--finetune" in cmd:
                assert(nums[9] == 2 or nums[10] == 1) # currently only finetuner
                ftpath = "007_{}_{}".format(model_name, id[:19]+('0'*len(id[19:])))
                cmd = cmd.replace("--finetune", pathstr.format("--finetune", ftpath))
            stuff = [(mod, merge(initial, kwargs)) for mod, initial in models]
            searchable[name] = (cmd, [kws for _, kws in stuff])
            experiments[name] = (cmd.split(), [lambda a, tmp=kws: mod(a, **tmp) for mod, kws in stuff])
            if name == args.run:
                return

if args.run is None or args.run not in experiments:
    add_experiments_007()

if __name__ == '__main__':
    if args.run is not None:
        cmdline, models = experiments[args.run]
        nargs = setup(add_common_arguments(argparse.ArgumentParser()), add_default_cmdline(cmdline))
        cmpobj = ModelCompare(nargs, [x(nargs) for x in models])
        if args.dump_npz:
            for model in cmpobj.models:
                output_name = os.path.join(cmpobj.args.save, model.model_name + ".npz")
                params = {name: param.data.cpu().numpy() for name, param in model.named_parameters()}
                np.savez(output_name, **params)
        else:
            cmpobj.run()
    elif args.exp is not None:
        expset = exp_set[args.exp]
        for cmdlinearg, filter in args.filter:
            tmp = {}
            for name, arg in expset.items():
                if cmdlinearg == "CMD":
                    if filter in arg[0]:
                        tmp[name] = arg
                if cmdlinearg == "NAME":
                    if filter in name:
                        tmp[name] = arg
                else:
                    for kwargs in arg[1]:
                        if filter in str(kwargs.get(cmdlinearg, "")):
                            tmp[name] = arg
            expset = tmp
        expset = list(expset.items())
        if args.random:
            random.shuffle(expset)
        if args.top > 0:
            expset = expset[:args.top]
        if args.ls:
            for name, arguments in expset:
                if args.print:
                    print(name, arguments)
                else:
                    print(name)
        elif args.validate is not None:
            names = set(x[0] for x in expset)
            with open(args.validate, 'r') as f:
                inputs = [x.strip() for x in f.readlines()]
            for x in inputs:
                if x not in names:
                    print(x)
