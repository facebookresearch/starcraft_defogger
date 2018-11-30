# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import sys, time, tempfile, os, copy, random
from os import path
start_time = time.time()
import logging
import warnings
from types import SimpleNamespace
warnings.filterwarnings("ignore", "", DeprecationWarning)

from data import DataLoader, ChainedDataLoader
from conv_lstm_utils import get_featurize_fn
from utils import State, UnitTypesMapper as utm, Timer
from collections import defaultdict, deque
import _ext


HuberLoss = th.nn.SmoothL1Loss()

def avg(l):
    return sum(l)/len(l)


class DummyModel(object):
    def __init__(self, mname):
        self.model_name = mname


def compute_prec_recall(pred, gold):
    """ returns an average prediction and recall for pred and gold """
    assert pred.shape == gold.shape, "prediction and gold don't have the same dimensions"
    assert pred.dtype == 'bool'
    assert gold.dtype == 'bool'
    true_pos = (pred * gold).sum()
    psum = pred.sum()
    gsum = gold.sum()
    prec = 0 if psum == 0 else true_pos / psum
    recall = 0 if gsum == 0 else true_pos / gsum
    return np.array([prec, recall])


def compute_dist_units(pred, gold):
    """ returns an average distance for each unit """
    # pred[frame, unittype, x, y]
    # TODO Hungarian (scipy.optimize.linear_sum_assignment)? How do we deal with dead units? Right now hack to have an approx of the avg bias
    assert pred.shape == gold.shape, "prediction and gold don't have the same dimensions"
    pred = pred - gold.min()
    gold = gold - gold.min()
    p_x = np.average(pred, axis=2, weights=np.arange(pred.shape[2]))
    p_y = np.average(pred, axis=3, weights=np.arange(pred.shape[3]))
    x = np.average(gold, axis=2, weights=np.arange(gold.shape[2]))
    y = np.average(gold, axis=3, weights=np.arange(gold.shape[3]))
    #return np.mean(np.sqrt(((y - p_y)**2).mean(-1) + ((x - p_x)**2).mean(-1)))
    ret = float(
        np.sum(np.sqrt(((y - p_y)**2).sum(-1) + ((x - p_x)**2).sum(-1))) /
        (gold.sum() + 1E-9))
    return ret


def hmap(vis, name, nparr, opts):
    o = {'title': name}
    o.update(opts)
    vis.heatmap(nparr, opts=o)


def bar(vis, name, t, p, select=False):
    assert t.shape[0] == p.shape[0], "want to plot histograms of different lengths"
    if select:
        nonnulls = t != 0
        # visdom doesn't let us plot 1x2 arrays :-/
        if nonnulls.sum() == 1:
            nonnulls[0] = True
            nonnulls[1] = True
        tmp = np.zeros((nonnulls.sum(), 2))
        tmp[:, 0] = t[nonnulls]
        tmp[:, 1] = p[nonnulls]
        vis.bar(tmp, opts={'title': name, 'legend': ['true', 'predicted']})
    else:
        tmp = np.zeros((t.shape[0], 2))
        tmp[:, 0] = t
        tmp[:, 1] = p
        vis.bar(
            tmp,
            opts={
                'title': name,
                'legend': ['true', 'predicted'],
                'rownames': np.where(nonnulls)
            })


def register_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        if any(
                np.any(np.isnan(gi.data.cpu().numpy())) for gi in grad_input
                if gi is not None):
            raise RuntimeError('NaN gradient in ' + type(module).__name__)

    def check_output(module, input, output):
        # print(module) you can add this to see that the hook is called
        if type(output) != tuple:
            output = (output, )
        if any(
                np.any(np.isnan(o.data.cpu().numpy())) for o in output
                if o is not None):
            raise RuntimeError('NaN output in ' + type(module).__name__)

    model.apply(lambda module: module.register_backward_hook(check_grad))
    model.apply(lambda module: module.register_forward_hook(check_output))


REGRESSION = 0
UNIT_CLASS = 1
BLDG_CLASS = 2
OPBT_CLASS = 3

class ModelCompare(object):
    def __init__(self, args, models=None):
        loss_fns = []
        self.update_number = 0
        for model in models:
            logging.info("Model: {}".format(model))
            loss_fns.append(self.__mk_loss_fn(args))
            # TODO a structured prediction (local and or global) type of loss?
            # TODO a mass conservation (except where we have factories?) loss?
            # TODO the loss/model should make you pay less for small (dx, dy) in prediction

            if args.check_nan:
                register_nan_checks(model)

        def xfer(tensor):
            if args.gpu >= 0:
                return tensor.cuda(args.gpu)
            return tensor

        models = [xfer(model) for model in models]

        self.args = args
        self.xfer = xfer
        self.models = models
        self.loss_fns = loss_fns
        self.train_dl = self.__get_dataloader()
        self.other_dl = self.__get_dataloader(both=False)
        self.featurizer = args.featurizer
        self.optimizers = []

        self.valid_every = args.valid_every
        self.plot_loss_every = 100
        self.n_input_timesteps = args.n_input_timesteps
        self.train_loss_pane = None

        self.save_timer = time.time()

        if args.load is not False and path.exists(args.load) and \
           path.exists(path.join(args.load, models[0].model_name + ".pth")):
            logging.info("Loading model from {}".format(args.load))
            self.load(args, models, args.load)
        if args.finetune != "" and path.exists(args.finetune) and \
           path.exists(path.join(args.finetune, models[0].model_name + ".pth")):
            logging.info("finetuneing model from {}".format(args.finetune))
            self.load(args, models, args.finetune)
        else:
            logging.info("No previous model found, initting new models")
            self.init(args, models)

        for model in models:
            nparam = sum([param.data.numel() for param in model.parameters()])
            logging.log(42, "n_param {} {}".format(model.model_name, nparam))

        args.featurizer = None  # can't pickle


    def __mk_loss_fn(self, args):
        loss_fn_dict = {
            "MSE": th.nn.MSELoss,
            "SmoothL1": th.nn.SmoothL1Loss,
            "SoftMargin": th.nn.MultiLabelSoftMarginLoss,
            "KL": th.nn.KLDivLoss  # TODO (the target in training)
        }
        reg_loss = loss_fn_dict.get(args.loss, th.nn.__dict__.get(args.loss, None))(size_average=False)
        def loss(inp, output, targ):
            # TODO becareful about size_averaging and BPTT
            loss = args.regression_lambda * reg_loss(output[0], targ)
            if self.state.delta:
                targ = targ + inp
            existence = targ > 1e-3 / args.divide_by

            if args.unit_loss_lambda > 0:
                our_units = existence[:, utm.our_units_inds, :, :]
                nmy_units = existence[:, utm.nmy_units_inds, :, :]
                units = th.cat([our_units, nmy_units], dim=1).float()
                uloss = th.nn.functional.binary_cross_entropy_with_logits(
                    output[UNIT_CLASS], units, size_average=False)
                loss += args.unit_loss_lambda * uloss


            if args.bldg_loss_lambda > 0:
                our_bldgs = existence[:, utm.our_bldgs_inds, :, :]
                nmy_bldgs = existence[:, utm.nmy_bldgs_inds, :, :]
                bldgs = th.cat([our_bldgs, nmy_bldgs], dim=1).float()
                bloss = th.nn.functional.binary_cross_entropy_with_logits(
                    output[BLDG_CLASS], bldgs, size_average=False)
                loss += args.bldg_loss_lambda * bloss

            if args.opbt_loss_lambda > 0:
                nmy_bldgs = existence[:, utm.nmy_bldgs_inds, :, :].sum(dim=3).sum(dim=2)
                nmy_bldgs = nmy_bldgs.float() > 1e-3 / args.divide_by
                bloss = th.nn.functional.binary_cross_entropy_with_logits(
                    output[OPBT_CLASS], nmy_bldgs.float(), size_average=False)
                loss += args.opbt_loss_lambda * bloss


            # TODO Consistency loss between regression and classification heads?
            return loss

        return loss


    def __get_dataloader(self, both=True):
        def post(x):
            scmap, race, inp, targ, game_name, vis = x
            inp, targ = self.view_input_target(inp, targ)
            x = scmap, race, inp, targ
            x = [self.xfer(Variable(i, requires_grad=False)) for i in x]
            x.append(game_name)
            x.append(vis)
            return x
        dl0 = DataLoader(self.args, get_featurize_fn(self.args, 0), post)
        dl1 = DataLoader(self.args, get_featurize_fn(self.args, 1), post)
        # Chaining remove correlations from training on one side (player) and the other in the same minibatch
        if both:
            return ChainedDataLoader(dl0, dl1)
        else:
            return dl0

    def init(self, args, models):
        ''' Instead of load '''
        optimizers = []
        for model in models:
            optim_fn = optim.__dict__[args.optim]
            kwargs = {
                'lr': args.lr,
                'weight_decay': args.lr * args.weight_decay
            }
            if optim_fn == optim.SGD or optim_fn == optim.RMSprop:
                kwargs['momentum'] = args.momentum
            optimizers.append({'model': optim_fn(model.parameters(), **kwargs)})
            if hasattr(model, "with_z") and model.with_z:
                optimizers[-1]['zbwd'] = model.z_opt([model.zbwd], lr=model.z_lr)

        self.optimizers = optimizers

        # Just stuff we want to keep between runs
        self.state = State({
            # bookeeping
            'epoch': 0,
            'running_loss': [[] for _ in models],
            'running_valids': [[] for _ in models],
            'running_f1_scores': [defaultdict(list) for _ in models],
            'best_valid': [1.0E10 for _ in models],
            'n_samples': 0,
            'n_frame': 0,

            # parsed arguments
            'only_us': args.predict == "only_us",
            'full_info': args.predict == "full_info",
            'only_defog': args.predict == "only_defog",
            'delta_full_info': False,
            'delta': args.predict_delta,

            # Time between reports, so we know how long things took
            'c_frames': 0,
            'c_time': start_time,
            'args': args,
            'version': 0,  # increment this when stuff changes
        })
        # The state object gives us the . operator and stops us from
        # writing to spurious arguments by accident

    def load(self, args, models, loadloc):
        self.optimizers = []
        for mi, model in enumerate(self.models):
            dic = th.load(path.join(loadloc, model.model_name + ".pth"))
            self.state = dic['state']
            optim_fn = optim.__dict__[self.state.args.optim]
            kwargs = {
                'lr': self.state.args.lr,
                'weight_decay': self.state.args.lr * self.state.args.weight_decay
            }
            if optim_fn == optim.SGD or optim_fn == optim.RMSprop:
                kwargs['momentum'] = self.state.args.momentum
            self.optimizers.append({'model': optim_fn(model.parameters(), **kwargs)})
            opt_st_dict = dic['optimizer_state_dict']
            with th.cuda.device(self.args.gpu):  # TODO wrap everything?
                self.optimizers[-1]['model'].load_state_dict(opt_st_dict['model'])
                if hasattr(model, "with_z") and model.with_z:
                    self.optimizers[-1]['zbwd'] = model.z_opt([model.zbwd], lr=model.z_lr)
                    self.optimizers[-1]['zbwd'].load_state_dict(opt_st_dict['zbwd'])
            self.state.args = args
            model.load_state_dict(dic['model_state_dict'])

    def run(self):
        try:
            if not self.args.just_valid:
                self.train()
                self.run_test()
            else:
                self.run_valid(0)
        except KeyboardInterrupt:
            if self.args.debug < 2:
                import pdb
                pdb.set_trace()
        self.train_dl.stop()
        self.other_dl.stop()



    def print_losses(self, lvl):
        lar = np.array(self.state.running_loss)
        nlar = lar[:, -self.plot_loss_every:].mean(axis=1)
        for mi, model in enumerate(self.models):
            logging.log(lvl, "tr_loss running_avg {} {}".format(
                model.model_name.ljust(20),
                nlar[mi]))

    def print_time(self, lvl):
        state = self.state
        dtime = time.time() - state.c_time
        if state.n_frame < state.c_frames:
            state.c_frames = 0
        dframes = state.n_frame - state.c_frames
        logging.log(
            lvl, "time {} frames {} fps_since_last_reported {} fps_total {} ".
            format(dframes, dframes / dtime, dtime,
                   state.n_frame / (time.time() - start_time)))
        state.c_time = time.time()
        state.c_frames = state.n_frame

    def __generate_bptt_slices(self, max):
        slices = range(0, max, self.args.bptt
                       if self.args.bptt > 0 else 100000)
        slices = list(slices) + [max]
        return zip(slices[:-1], slices[1:])

    def __do_losses(self, output, targ):
        pass

    def __init_metrics(self):
        self.baselines = [
            'baseline_input', 'baseline_memory', # 'baseline_input_rules',
            'baseline_memory_rules', 'baseline_mem_prev', # 'baseline_mem_max',
        ]
        # TODO: move other_metrics to self.state / init, but only once we stabilize what's in it
        other_metrics = [
            {
                'op_bt': np.zeros(2),  # measures our prediction of their build tree [prec, recall]
                'hid_u': np.zeros(2),  # measures our prediction of their hidden units (mostly makes sense in defogging) [prec, recall]
                'nmy_u': np.zeros(2),  # measures our prediction of their hidden units (mostly makes sense in defogging) [prec, recall]
                # Same things based off of the classification loss
                'p_op_bt': np.zeros(2), # Pooled loss designed for op_bt
                '0_L1': 0,
                '1_L1': 0,
                '2_L1': 0,
                '0_SmoothL1': 0,
                '1_SmoothL1': 0,
                '2_SmoothL1': 0,
            } for _ in self.models + self.baselines
        ]
        return other_metrics

    def __accumulate_metrics(self, inp, targ, vis, outputs, other_metrics):
        threshold = 0.1 / self.args.divide_by  # 10% of the value when 1
        inp_np = inp.cpu().data.clone().numpy()  # clone() in case we're on CPU already
        targ_np = targ.cpu().data.clone().numpy()  # clone() in case we're on CPU already
        hid = 1 - vis
        inp_hid, targ_hid = self.view_input_target(hid[:-1], hid[1:])
        if self.state.delta:
            targ_np = inp_np + targ_np
            targ_hid = targ_hid + inp_hid
        inp_hid = inp_hid > 0
        targ_hid = targ_hid > 0
        inf_mask = inp_hid

        assert (inp_np < 0).sum() == 0, "Input should be all positive numbers..."
        assert (targ_np < 0).sum() == 0, "Input should be all positive numbers..."
        inp_our_units = inp_np[:, utm.our_units_inds, :, :]
        inp_nmy = inp_np[:, utm.offset:]
        inp_nmy_units = inp_np[:, utm.nmy_units_inds, :, :]
        inp_nmy_bldgs = inp_np[:, utm.nmy_bldgs_inds, :, :]
        inp_sum = inp_np.sum(-1).sum(-1)
        inp_our_sum = inp_sum[:, :utm.offset]
        inp_nmy_sum = inp_sum[:, utm.offset:]
        inp_our_bldgs_sum = inp_sum[:, utm.our_bldgs_inds]
        inp_nmy_bldgs_sum = inp_sum[:, utm.nmy_bldgs_inds]
        our_units = targ_np[:, utm.our_units_inds, :, :]
        nmy_units = targ_np[:, utm.nmy_units_inds, :, :]
        nmy_bldgs = targ_np[:, utm.nmy_bldgs_inds, :, :]
        nmy_hidden_units = (nmy_units * inf_mask) > 0
        nmy_hidden_bldgs = (nmy_bldgs * inf_mask) > 0
        targ_sum = targ_np.sum(-1).sum(-1)
        our_bldgs_sum = targ_sum[:, utm.our_bldgs_inds]
        nmy_bldgs_sum = targ_sum[:, utm.nmy_bldgs_inds]
        gold_op_bt = (nmy_bldgs_sum > threshold)
        #nmy_units_sum = targ_sum[:, utm.nmy_units_inds]
        nmy_hidden_whole = (nmy_units * inf_mask) * self.args.divide_by
        e_nmy_units = (nmy_units > 0)

        hid_true_positives = []

        for i, output in enumerate(outputs):
            output_np = output[REGRESSION].cpu().clone().numpy()
            # unit_output_np = output[UNIT_CLASS].cpu().clone().numpy()
            # bldg_output_np = output[BLDG_CLASS].cpu().clone().numpy()
            opbt_output_np = output[OPBT_CLASS].cpu().clone().numpy()
            if self.state.delta:
                output_np += inp_np
            p_our_units = output_np[:, utm.our_units_inds, :, :]
            p_nmy_units = output_np[:, utm.nmy_units_inds, :, :]
            # p_nmy_bldgs = output_np[:, utm.nmy_bldgs_inds, :, :]
            output_sum = output_np.sum(-1).sum(-1)
            # p_our_bldgs_sum = output_sum[:, utm.our_bldgs_inds]
            p_nmy_bldgs_sum = output_sum[:, utm.nmy_bldgs_inds]
            #p_nmy_units_sum = output_sum[:, utm.nmy_units_inds]

            logit = lambda x: np.log(x) - np.log(1-x)
            if self.args.class_prob_thresh >= 1:
                class_thresh = 1e100
            elif self.args.class_prob_thresh <= -1:
                class_thresh = -1e100
            else:
                class_thresh = logit(self.args.class_prob_thresh)

            our_res = other_metrics[i]

            our_res['op_bt'] += compute_prec_recall(p_nmy_bldgs_sum > self.args.n_unit_thresh, gold_op_bt)
            p_nmy_units_e = p_nmy_units > self.args.n_unit_thresh
            our_res['hid_u'] += compute_prec_recall(p_nmy_units_e * inf_mask, nmy_hidden_units)
            our_res['nmy_u'] += compute_prec_recall(p_nmy_units_e, nmy_units > 0)
            hid_true_positives.append((p_nmy_units_e * inf_mask * nmy_hidden_units) > 0)

            tmp = p_nmy_units * self.args.regr_slope_scalar - self.args.n_unit_thresh
            our_res[str(i) + '_L1'] += abs(tmp - nmy_units).sum()
            our_res[str(i) + '_SmoothL1'] += HuberLoss(Variable(th.from_numpy(tmp), requires_grad=False), Variable(th.from_numpy(nmy_units), requires_grad=False)).data[0]
            our_res[str(i)+'_L1'] += abs((p_nmy_units) - nmy_units).sum()
            our_res['p_op_bt'] += compute_prec_recall(opbt_output_np > class_thresh, gold_op_bt)

        # baselines
        end = len(outputs)
        bl_inp = other_metrics[end + 0]
        bl_mem = other_metrics[end + 1]
        bl_mem_r = other_metrics[end + 2]
        bl_mem_p = other_metrics[end + 3]

        # baseline_input
        bl_inp['op_bt'] += compute_prec_recall(inp_nmy_bldgs_sum > threshold, gold_op_bt)
        bl_inp['hid_u'] += compute_prec_recall((inp_nmy_units * inf_mask) > threshold, nmy_hidden_units)
        for i, _ in enumerate(outputs):
            bl_inp[str(i) + '_L1'] += abs(inp_nmy_units - nmy_units).sum()
            bl_inp[str(i) + '_SmoothL1'] += HuberLoss(Variable(th.from_numpy(inp_nmy_units), requires_grad=False), Variable(th.from_numpy(nmy_units), requires_grad=False)).data[0]

        bl_inp['p_op_bt']  = bl_inp['op_bt']


        # baseline_memory
        max_our_bldgs_sum = np.maximum.accumulate(inp_our_bldgs_sum)
        max_nmy_bldgs_sum = np.maximum.accumulate(inp_nmy_bldgs_sum)
        max_nmy_bldgs = np.maximum.accumulate(inp_nmy_bldgs)
        max_nmy_units = np.maximum.accumulate(inp_nmy_units)
        max_our_sum = np.maximum.accumulate(inp_our_sum)
        max_nmy_sum = np.maximum.accumulate(inp_nmy_sum)
        max_nmy = np.maximum.accumulate(inp_nmy)
        bl_mem['op_bt'] += compute_prec_recall(max_nmy_bldgs_sum > threshold, gold_op_bt)
        bl_mem['hid_u'] += compute_prec_recall((max_nmy_units * inf_mask) > threshold, nmy_hidden_units)
        bl_mem['nmy_u'] += compute_prec_recall(max_nmy_units > threshold, nmy_units > 0)
        for i, _ in enumerate(outputs):
            bl_mem[str(i) + '_L1'] += abs(max_nmy_units - nmy_units).sum()
            bl_mem[str(i) + '_SmoothL1'] += HuberLoss(Variable(th.from_numpy(max_nmy_units), requires_grad=False), Variable(th.from_numpy(nmy_units), requires_grad=False)).data[0]

        bl_mem['p_op_bt']  = bl_mem['op_bt']


        # baseline_memory_rules
        inp_nmy_mem_r = utm.ttrules(max_nmy.reshape(-1, max_nmy.shape[1]) > threshold).reshape(max_nmy.shape)
        bl_mem_r['op_bt'] += compute_prec_recall(utm.ttrules(max_nmy_sum > threshold)[:, utm.our_bldgs_inds], gold_op_bt)
        bl_mem_r['hid_u'] += compute_prec_recall(inp_nmy_mem_r[:, utm.our_units_inds, :, :] * inf_mask > threshold, nmy_hidden_units)  # fishy
        bl_mem_r['nmy_u'] += compute_prec_recall(inp_nmy_mem_r[:, utm.our_units_inds, :, :] > threshold, nmy_units > 0)
        bl_mem_r['0_L1'] += -1
        bl_mem_r['1_L1'] += -1
        bl_mem_r['0_SmoothL1'] += -1
        bl_mem_r['1_SmoothL1'] += -1

        bl_mem_r['p_op_bt']  = bl_mem_r['op_bt']

        inp_mem_prev = inp_np.copy()
        for i in range(1, inp_np.shape[0]):
            inp_mem_prev[i] = (1 - inp_hid[i]) * inp_np[i] + inp_hid[i] * inp_mem_prev[i-1]
        mem_prev_nmy_units = inp_mem_prev[:, utm.nmy_units_inds, :, :]
        mem_prev_nmy_bldgs = inp_mem_prev[:, utm.nmy_bldgs_inds, :, :]
        mem_prev_our_bldgs_sum = inp_mem_prev[:, utm.our_bldgs_inds, :, :].sum(-1).sum(-1)
        mem_prev_nmy_bldgs_sum = inp_mem_prev[:, utm.nmy_bldgs_inds, :, :].sum(-1).sum(-1)

        bl_mem_p['op_bt'] += compute_prec_recall(mem_prev_nmy_bldgs_sum > threshold, gold_op_bt)
        bl_mem_p['hid_u'] += compute_prec_recall((mem_prev_nmy_units * inf_mask) > threshold, nmy_hidden_units)
        bl_mem_p['nmy_u'] += compute_prec_recall(mem_prev_nmy_units > threshold, nmy_units > 0)
        for i, _ in enumerate(outputs):
            bl_mem_p[str(i) + '_L1'] += abs(mem_prev_nmy_units - nmy_units).sum()
            bl_mem_p[str(i) + '_SmoothL1'] += HuberLoss(Variable(th.from_numpy(mem_prev_nmy_units), requires_grad=False), Variable(th.from_numpy(nmy_units), requires_grad=False)).data[0]

        bl_mem_p['p_op_bt']  = bl_mem_p['op_bt']


    def _run_inference(self, msg, n_games=0):
        other_metrics = self.__init_metrics()
        n_f = 0

        tot_loss = [0 for _ in self.models]
        n_g = 0
        n_f = 0
        scmap, race, inp, targ = 0, 0, 0, 0
        for model in self.models:
            model.eval()
        for data in self.other_dl:
            if n_games > 0 and n_g >= n_games:
                break
            n_g += 1
            scmap, race, inp, targ, _, vis = data

            outputs, losses = self._do_model_step(scmap, race, inp, targ, None, optimize=False)
            n_f += targ.size(0)

            for mi, loss in enumerate(losses):
                tot_loss[mi] += loss * (targ.size(0) if self.args.loss_averaging else 1)

            self.__accumulate_metrics(inp, targ, vis, outputs, other_metrics)

        logging.log(42, msg)
        ret = defaultdict(list)
        ret['outputs'] = [[y.cpu() for y in x] for x in outputs]
        ret['inputs'] = (scmap.cpu(), race.cpu(), inp.cpu(), targ.cpu())
        ret['loss'] = [(tl / n_f) for tl in tot_loss]
        ret['metrics'] = defaultdict(list)
        for i, model in enumerate(self.models + [DummyModel(bname) for bname in self.baselines]):
            for metric_key, values in other_metrics[i].items():
                # values are normalized per game, n_f would normalize per frame
                v = values / n_g
                mk = metric_key
                if not np.isscalar(v):
                    mk = mk + '_prf'
                    v = [v[0], v[1], 2 * v[0] * v[1] / (v[0] + v[1] + 1E-9)]  # TODO F1 score after all averaging ok?
                    format = "{:<10.4f}"
                else:
                    v = [v]
                    format = "{:<10.4E}"
                if type(model) != DummyModel:
                    ret['metrics'][mk].append(v)
                logging.log(42, "{} {} {}".format(
                    model.model_name.ljust(20),
                    metric_key.ljust(10), " ".join(format.format(x) for x in v)))
            if type(model) != DummyModel:
                model.train()
        return dict(ret)


    def run_valid(self, n_valid=1000):
        state = self.state
        args = self.args
        self.print_losses(42)
        self.print_time(42)
        self.other_dl.valid()
        logging.info("Running validation...")
        valid_ret = self._run_inference(
            "validation after_n_samples {} with_n_valid {}".format(
                self.state.n_samples, n_valid), n_valid)

        lar = np.array(state.running_loss)
        for mi, model in enumerate(self.models):
            if args.save:  # TODO save on other metrics?
                logging.log(42, "saving {}".format(model.model_name))
                self.save(model, self.optimizers[mi])
                if valid_ret['loss'][mi] < state.best_valid[mi]:
                    self.save(model, self.optimizers[mi], "_best.pth")
                    state.best_valid[mi] = valid_ret['loss'][mi]
            if args.lr_decay and state.n_samples > 10000 or self.args.small:
                logging.info("Doing LR decay")
                recent_mean_loss = lar[mi][-400:].mean()
                older_mean_loss = lar[mi][-1400:-1000].mean()
                if recent_mean_loss > 0.99 * older_mean_loss or self.args.small:  # TODO with lr_decay only
                    with th.cuda.device(self.args.gpu):  # TODO wrap everything?
                        sd = self.optimizers[mi]['model'].state_dict()
                        sd['param_groups'][0]['lr'] = max(
                            sd['param_groups'][0]['lr'] * 0.5, 1E-8)
                        logging.log(42, "lr_decay {}".format(sd['param_groups'][0][
                            'lr']))
                        self.optimizers[mi]['model'].load_state_dict(sd)
            model.train()

        # TODO plot debug outputs on the valid set
        # if args.debug < 3:
        #     th.save(self.models, tempfile.gettempdir()+ '/models_' + str(int(start_time)) + '.pth')
        self.__update_plots(valid_ret)

    def run_test(self):
        self.other_dl.test()
        logging.info("Running test...")
        self._run_inference("test", n_games=0)

    def view_input_target(self, inp, targ):
        # default is defog, as inp is with FoW at t and targ without at t+1
        state = self.state
        if state.full_info:
            inp = targ[:-1]
            targ = targ[1:]
        elif state.only_us:
            targ = inp[1:]
            inp = inp[:-1]
        elif state.delta_full_info: # TODO add this option, TODO change test time!
            targ = targ[1:] - targ[:-1]
            inp = inp[1:]
        elif state.only_defog:
            inp = inp[1:]
            targ = targ[:-1]
        if state.delta:
            targ = targ - inp
        if self.n_input_timesteps > 1:  # TODO
            # inp = stack inp # TODO
            targ = targ[1:]
        return inp, targ

    def _if_plot(func):
        def only_if_plot(self, *args, **kwargs):
            if self.args.plot is not None:
                func(self, *args, **kwargs)

        return only_if_plot

    @_if_plot
    def __init_graphs(self):
        import visdom
        if self.args.save:
            self.env_id = 'xp' + self.args.save
        else:
            self.env_id = 'main'
        self.vis = visdom.Visdom(server=self.args.plot, env=self.env_id)
        self.vis.text(self.args.__str__())
        self.train_loss_pane = self.vis.line(
            np.vstack([np.array([0, 0]) for _ in self.models]).transpose())
        self.valid_loss_pane = self.vis.line(
            np.vstack([np.array([0, 0]) for _ in self.models]).transpose())
        self.valid_F1_pane = self.vis.line(
            np.vstack([np.array([0, 0]) for _ in self.models]).transpose())
        self.valid_dist_pane = self.vis.line(
            np.vstack([np.array([0, 0]) for _ in self.models]).transpose())
        self.plotted_at_least_once = False

    @_if_plot
    def __update_train_graphs(self):
        lar = np.array(self.state.running_loss)
        if self.plotted_at_least_once:
            nlar = lar[:, -self.plot_loss_every:].mean(axis=1, keepdims=True)
            self.vis.line(
                np.log10(nlar.T),
                np.asarray(
                    [lar.shape[1]] * nlar.shape[0]).reshape(nlar.shape).T,
                win=self.train_loss_pane,
                update='append')
        else:
            round = (
                lar.shape[1] // self.plot_loss_every) * self.plot_loss_every
            lar = lar[:, -round:]
            nlar = lar.reshape(lar.shape[0], -1, self.plot_loss_every).mean(
                axis=2)
            self.vis.line(
                np.log10(nlar.transpose()),
                np.arange(lar.shape[1], step=self.plot_loss_every),
                win=self.train_loss_pane,
                opts={
                    'legend': [m.model_name for m in self.models],
                    'title': "(log10) Training loss",
                })
            self.plotted_at_least_once = True

    def __plot_heatmaps(self, last_outs, inp, targ, size, msg=""):
        n_g = self.state.n_samples

        inplt = inp.data.cpu().numpy()[size].mean(axis=0)
        targplt = targ.data.cpu().numpy()[size].mean(axis=0)
        modelplts = [(self.models[mi].model_name,
                      output[REGRESSION].cpu().numpy()[size].mean(axis=0))
                     for mi, output in enumerate(last_outs)]
        opts = {
            'xmin': float(min([min(v.min() for _, v in modelplts), inplt.min(), targplt.min()])),
            'xmax': float(max([max(v.max() for _, v in modelplts), inplt.max(), targplt.max()])),
        }
        hmap(self.vis, '{}: {}inp: {}'.format(n_g, msg, list(inp.size())), inplt, opts=opts)
        hmap(self.vis, '{}: {}targ: {}'.format(n_g, msg, list(targ.size())), targplt, opts=opts)
        for name, value in modelplts:
            hmap(self.vis, '{}: {}output for {}'.format(n_g, msg, name), value, opts=opts)

    @_if_plot
    def __update_plots(self, valid_ret):
        ''' valid_ret is the dictionary returned by a _run_inference '''
        logging.info("Updating validation plots...")

        '''
        fixed_gn = "18/TL_PvZ_GG5758.tcr" + ("" if not self.args.reduced else ".npz")
        scmap, race, inp, targ, _, _ = self.other_dl.one(fixed_gn)
        last_outs, _ = self._do_model_step(scmap, race, inp, targ, None, optimize=False)
        # Replay is 10960 frames long (8fps)
        frame = min(inp.data.cpu().numpy().shape[0] - 1,
                    5000 // self.args.combine_frames)
        self.__plot_heatmaps(last_outs, inp, targ, frame, "fxd ")
        '''

        scmap, race, inp, targ = valid_ret['inputs']
        last_outs = valid_ret['outputs']
        length = inp.data.cpu().numpy().shape[0]
        self.__plot_heatmaps(last_outs, inp, targ, length // 2)

        '''
        [hmap(self.vis, 'encoded hidden for {}'.format(model.model_name),
            model.encoder(model.conv1x1(model.trunk(scmap, race, inp))
                .contiguous()).data.cpu().numpy()[length//2]
                    .reshape(model.enc_embsize, -1))
            for model in self.models]
        '''
        n_g = self.state.n_samples
        for model, output in zip(self.models, last_outs):
            bar(self.vis,
                '{}: units by type for {}'.format(n_g, model.model_name),
                targ.data.cpu().numpy()[length // 2].sum(axis=2).sum(axis=1),
                output[REGRESSION].cpu().numpy()[length // 2].sum(axis=2).sum(axis=1),
                select=True)

        for i, v in enumerate(valid_ret['loss']):
            self.state.running_valids[i].append(v)
        lar = np.array(self.state.running_valids)  # lar[model_id][valid_time]
        self.vis.line(
            np.log10(lar.transpose()),
            np.arange(lar.shape[1]),
            win=self.valid_loss_pane,
            opts={
                'legend': [m.model_name for m in self.models],
                'title': '(log10) Validation loss',
            })
        lar_F1 = []  # lar_F1[F1score*model_id][valid_time]
        legend_F1 = []
        lar_dist = []
        legend_dist = []
        for metric, valids in valid_ret['metrics'].items():
            for i, v in enumerate(valids):
                if '_L1' in metric:
                    lar = lar_dist
                    legend = legend_dist
                else:
                    lar = lar_F1
                    legend = legend_F1
                self.state.running_f1_scores[i][metric].append(v[-1])
                lar.append(self.state.running_f1_scores[i][metric])
                legend.append(self.models[i].model_name + "_" + metric)
        lar_F1 = np.array(lar_F1)
        self.vis.line(
            lar_F1.transpose(),
            np.arange(lar_F1.shape[1]),
            win=self.valid_F1_pane,
            opts={
                'legend': legend_F1,
                'title': 'F1 validation scores',
            })
        lar_dist = np.array(lar_dist)
        self.vis.line(
            lar_dist.transpose(),
            np.arange(lar_dist.shape[1]),
            win=self.valid_dist_pane,
            opts={
                'legend': legend_dist,
                'title': 'dist validation scores',
            })
        self.vis.save([self.env_id])
        logging.info("Done updating validation plots")

    def train(self):
        self.__init_graphs()

        timer = time.time()
        while self.state.epoch < self.args.epochs:
            logging.log(42, "epoch {}".format(self.state.epoch))
            self.train_dl.train()
            for data in self.train_dl:
                self._do_train_step(data)
            self.state.epoch += 1

    def __do_model_step(self, mi, scmap, race, inp, targ, game_name, optimize=True):
        '''
        Single bptt step of the model
        '''
        model = self.models[mi]
        has_z = hasattr(model, "with_z") and model.with_z
        model.hidden = model.repackage_hidden(model.hidden)
        if optimize:
            for _, optimizer in self.optimizers[mi].items():
                optimizer.zero_grad()

        if has_z and optimize:
            old_hiddens = model.hidden
            should_retain = hasattr(model, "z_pred_cut_gradient") and not model.z_pred_cut_gradient

            if model.zbwd_init_zfwd:
                if not model.zbwd_single:
                    model.zbwd = Variable(th.zeros(inp.size(0),1,model.zsize).type(th.cuda.FloatTensor))
                model.zbwd_initialized = False

            # disable gradient updates to the rest of the model
            for param in model.parameters():
                param.requires_grad = False
            model.zbwd.requires_grad = True

            # TODO remove this multiple optimizer stuff since we just create a new one here
            self.optimizers[mi]['zbwd'] = model.z_opt([model.zbwd], lr=model.z_lr)
            if not hasattr(model, "tail_values"):
                model.tail_values = {'zbwd_norm': deque(maxlen=200), 'zbwd_grad_norm': deque(maxlen=200), 'zpred_norm': deque(maxlen=200), 'zpred_loss': deque(maxlen=200), 'zpred_grad_norm': deque(maxlen=200)}
            z_step = 0
            # Do a bunch of model forwards to calculate the gradient wrt z_bwd
            if model.zfwd_zbwd_ratio > 0 and random.random() > model.zfwd_zbwd_ratio:
                model.hidden = model.repackage_hidden(old_hiddens)
                input, embed = model.trunk_encode_pool(scmap, race, inp)
                while z_step == 0 or model.zbwd_to_convergence and z_step < 5*np.log10(self.update_number+1) and model.z_lr * model.zbwd.grad.norm().data[0] > 1e-4:
                    self.optimizers[mi]['zbwd'].zero_grad()
                    model.hidden = model.repackage_hidden(old_hiddens)
                    output = model.forward_rest(input, embed)
                    loss = self.loss_fns[mi](inp, output, targ)
                    if self.args.loss_averaging:
                        loss = loss / targ.size(0)
                    loss.backward()
                    if self.update_number % 2000 == 0:
                        logging.log(42, "    Lf_zbwd_grad {} zbwd_norm {}".format(model.zbwd.grad.norm().data[0], model.zbwd.norm().data[0]))
                    self.optimizers[mi]['zbwd'].step()
                    z_step += 1
                model.tail_values['zbwd_norm'].append(model.zbwd.norm().data[0])
                model.tail_values['zbwd_grad_norm'].append(model.zbwd.grad.norm().data[0])
            if self.update_number % 2000 == 0:
                logging.log(42, "AVG_Lf_zbwd_grad {} zbwd_norm {}".format(avg(model.tail_values['zbwd_grad_norm']), avg(model.tail_values['zbwd_norm'])))
                logging.log(42, "Lf_zbwd_grad {}, zbwd_norm {}".format(model.zbwd.grad.norm().data[0], model.zbwd.norm().data[0]))

            # Finally, update the whole model
            for param in model.parameters():
                param.requires_grad = True

            model.hidden = model.repackage_hidden(old_hiddens)
            output = model(scmap, race, inp)
            loss = self.loss_fns[mi](inp, output, targ)
            if self.args.loss_averaging:
                loss = loss / targ.size(0)


            if model.zbwd_init_zfwd:
                zvar = F.tanh(model.zbwd).detach()
                if model.zbwd_single:
                    zvar = zvar.expand(inp.size(0), 1, model.zsize)
            else:
                zvar = Variable(model.zbwd.data).expand((inp.size(0), 1, model.zsize))
            zloss = model.z_lambda * model.zlossfn(model.zfwd, zvar)
            zloss.backward(retain_graph=True)
            zpred_grad_norm = next(model.zpred.parameters()).grad.norm().data[0]
            zpred_norm = next(model.zpred.parameters()).norm().data[0]
            if self.update_number % 2000 < 50:
                logging.log(42, "loss {}".format(loss.data[0]))
                logging.log(42, "Lz_zpred_loss {} zpred_grad_norm {} zpred_params_norm {}".format(zloss.data[0], zpred_grad_norm , zpred_norm))
            model.tail_values['zpred_norm'].append(zpred_norm)
            model.tail_values['zpred_grad_norm'].append(zpred_grad_norm)
            model.tail_values['zpred_loss'].append(zloss.data[0])
            if self.update_number % 2000 == 0:
                logging.log(42, "AVG_Lz_zpred_loss {} zpred_grad_norm {}, zpred_params_norm {}".format(avg(model.tail_values['zpred_loss']), avg(model.tail_values['zpred_grad_norm']), avg(model.tail_values['zpred_norm'])))
        else:
            output = model(scmap, race, inp)
            loss = self.loss_fns[mi](inp, output, targ)
            if self.args.loss_averaging:
                loss = loss / targ.size(0)

        if optimize:
            loss.backward()
            if self.args.clip > 0:
                th.nn.utils.clip_grad_norm(model.parameters(), self.args.clip)
            self.optimizers[mi]['model'].step()

        return [x.detach() for x in output], loss.detach()

    def _do_model_step(self, scmap, race, inp, targ, game_name, optimize=True):
        ''' Returns:
            ([list of outputs for each model], [list of losses for each model])
            '''
        self.update_number += 1
        for model in self.models:
            if hasattr(model, "init_z"):
                model.init_z(game_name)
            model.hidden = model.init_hidden()

        losses = []
        outputs = []
        for mi, model in enumerate(self.models):
            if self.args.bptt > 0 and model.accepts_bptt:
                cur_losses = []
                cur_outputs = []
                for start, end in self.__generate_bptt_slices(inp.size(0)):
                    output, loss = self.__do_model_step(mi, scmap, race, inp[start:end], targ[start:end], game_name)
                    loss *= end-start if self.args.loss_averaging else 1
                    cur_losses.append(loss.data.cpu())
                    cur_outputs.append([x.data.cpu() for x in output])
                # Need to transpose last_outs and concat across time dim
                outputs.append([th.cat(x) for x in zip(*cur_outputs)])
                losses.append(th.cat(cur_losses).sum())
                if self.args.loss_averaging:
                    losses[mi] /= outputs[mi].size(0)
            elif self.args.bptt == 0 or not model.accepts_bptt:
                output, loss = self.__do_model_step(mi, scmap, race, inp, targ, game_name, optimize)
                outputs.append([x.data.cpu() for x in output])
                assert(loss.numel() == 1)
                losses.append(loss.data.cpu().sum())
            else:
                # TODO bptt sampled with cut ~ Bernouilli(hyperparam)
                self.quit_training("BPTT < 0 not implemented")
        return outputs, losses

    def _do_train_step(self, data):
        ''' Does a single game '''
        args = self.args
        state = self.state
        scmap, race, inp, targ, game_name, _ = data

        _, losses = self._do_model_step(scmap, race, inp, targ, game_name)

        #if args.loss == 'SoftMargin': // TODO KL
        # strictly incorrect, turning regression of pos. + numbers
        # into position and presence (of the units)
        #targ = targ.ceil().long()

        for mi, loss in enumerate(losses):
            self.state.running_loss[mi].append(loss)

        state.n_samples += 1
        state.n_frame += inp.size(0)
        if state.n_samples % self.valid_every == 0:
            self.run_valid(1000)
        elif state.n_samples % self.plot_loss_every == 0:
            self.print_losses(logging.INFO)
            self.__update_train_graphs()
        elif time.time() - self.save_timer > 5 * 3600 and args.save:  # 5 hours
            for model, op in zip(self.models, self.optimizers):
                logging.log(42, "timed saving {}".format(model.model_name))
                self.save(model, op)
            self.save_timer = time.time()

    def save(self, model, optimizer, suffix='.pth'):
        th.save({
            'state': self.state,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizer.items()},
        }, path.join(self.args.save, model.model_name + suffix))

    def quit_training(self, msg="quitting"):
        self.train_dl.stop()
        logging.error(msg)
        sys.exit(-1)
