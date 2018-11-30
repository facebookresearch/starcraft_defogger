# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging, sys
from functools import partial
from itertools import chain

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import optim, nn
from torch.autograd import Variable

from utils import UnitTypesMapper as utm

# TODO: add visibility feature

class GatedConvolution(nn.Module):
    def __init__(self, convtype, *args, **kwargs):
        super(GatedConvolution, self).__init__()
        self.conv = convtype(*args, **kwargs)
        self.gate = convtype(*args, **kwargs)
        # TODO? could be done with one conv and first half gating the second half

    def forward(self, x):
        # TODO: check that the gating is learned to be 0 on the last layer,
        # when predicting delta, so that we can have negative values
        return F.sigmoid(self.gate(x)) * self.conv(x)


def get_GatedConvolution(type):
    return partial(GatedConvolution, type)


def _transpose_convtype(convtype):
    mapping = [
        (nn.Conv1d, nn.ConvTranspose1d),
        (nn.Conv2d, nn.ConvTranspose2d),
        (nn.Conv3d, nn.ConvTranspose3d),
    ]
    mapping = dict(mapping + [(b,a) for a,b in mapping])
    return mapping[convtype]

class IdentityFn(nn.Module):
    def forward(self, x):
        return x

class ResidualModule(nn.Module):
    def __init__(self, convtype, nonlin, input_size, output_size, convsize, stride=1, padding=0):
        super(ResidualModule, self).__init__()
        assert output_size % 4 == 0, "The output of the residual layer is {}, not divisible by 4".format(output_size)
        interm_size = output_size // 4
        self.conv1 = convtype(input_size, interm_size, 1, bias=False)
        self.conv2 = convtype(interm_size, interm_size, convsize, stride,
                              padding=padding)  #bias=False)
        self.conv3 = convtype(interm_size, output_size, 1, bias=False)
        self.nonlin = nonlin
        self.downsample = IdentityFn()
        if stride != 1 or input_size != output_size:
            assert padding == (convsize-1) // 2, "working with non aligned layers in resnet"
            self.downsample = convtype(input_size, output_size, 1,
                                       stride=stride, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.nonlin(self.conv1(x))
        out = self.nonlin(self.conv2(out))
        out = self.conv3(out)
        return out + self.downsample(residual)

def residual_layer(convtype, nonlin):
    return partial(ResidualModule, convtype, nonlin)


def convnet(convsize_0,
            convsize,
            padding_0,
            padding,
            conv,
            non_lin,
            input_size,
            interm_size,
            output_size,
            depth=2,
            stride_0=1,
            stride=1):
    ret = nn.Sequential()
    ret.add_module(
        "conv_0",
        conv(input_size, interm_size, convsize_0, stride_0, padding=padding_0)
    )
    if non_lin is not None:
        ret.add_module("nonlin_0", non_lin)
    for i in range(1, depth):
        ret.add_module(
            "conv_{}".format(i),
            conv(interm_size, interm_size, convsize, stride, padding=padding)
        )
        ret.add_module("nonlin_{}".format(i), non_lin)
    ret.add_module("convnet_output_layer".format(),
                   conv(interm_size, output_size, 1))
    return ret


def simple_convnet(convsize,
                   padding,
                   conv,
                   non_lin,
                   input_size,
                   output_size,
                   depth=2,
                   stride=1):
    ''' Less control but easier to specify, depth is what you get '''
    ret = nn.Sequential()
    for i in range(0, depth-1):
        ret.add_module(
            "conv_{}".format(i),
            conv(input_size, input_size, convsize, stride, padding=padding)
        )
        ret.add_module("nonlin_{}".format(i), non_lin)
    ret.add_module("conv_{}".format(depth-1), conv(input_size, output_size, 1))
    return ret


def decoder(convsize_0, convsize):
    padding_0 = (int(convsize_0)-1)//2
    padding = (int(convsize)-1)//2
    return partial(convnet, convsize_0, convsize, padding_0, padding)


class _MapRaceFeaturize(nn.Module):
    '''
    This puts the StarCraft map at the same pooling (kernel_size and stride)
    as the features coming from the featurizer, and concatenates with inputs.
    '''
    def __init__(self, args, map_embsize=64, race_embsize=8):
        super(_MapRaceFeaturize, self).__init__()
        self.args = args
        self.map_embsize = map_embsize
        self.race_embsize = race_embsize
        nonlin = nn.ELU(inplace=True)
        # TODO Make the sizes line up with resnets...
        self.map_model = nn.Sequential(
            nn.Conv2d(4, map_embsize, 4, 2, padding=1),
            nonlin,
            nn.Conv2d(map_embsize, map_embsize, args.kernel_size//2, args.stride//2),
            nonlin,
            nn.Conv2d(map_embsize, map_embsize, 3, padding=1),
        )
        self.race_embedding = nn.Embedding(3, race_embsize)

    def forward(self, scmap, race, input):
        bsz = input.size(0)
        H = input.size(2)
        W = input.size(3)
        map_features = self.map_model(scmap)  # 8 x H x W
        race_feature = self.race_embedding(race)  # 1 x 2 x R=race_embsize
        map_features = map_features.expand(bsz, self.map_embsize, H, W)
        race_feature = (race_feature
                        .view(1, self.race_embsize * 2, 1, 1)
                        .expand(bsz, self.race_embsize * 2, H, W)
                        )
        return th.cat([input, map_features, race_feature], dim=1)


class lstm(nn.Module):
    def __init__(self, args,
                 model_name='lstm',
                 map_embsize=64,
                 race_embsize=8,
                 residual=False,
                 dec_convsize=3,
                 dec_depth=3,
                 dec_embsize=128,
                 conv=nn.Conv2d,
                 nonlin=nn.ELU,
                 hid_dim=256,
                 rnn_input_size=2048,
                 lstm_nlayers=3,
                 lstm_dropout=0):
        super(lstm, self).__init__()
        self.model_name = model_name
        self.nonlin = nonlin(inplace=True) if nonlin is not None else IdentityFn()
        self.convmod = residual_layer(conv, self.nonlin) if residual else conv

        self.residual = residual
        self.hid_dim = hid_dim
        self.rnn_input_size = rnn_input_size
        self.lstm_num_layers = lstm_nlayers
        self.lstm_dropout = lstm_dropout
        self.dec_convsize = dec_convsize
        self.dec_depth = dec_depth
        self.dec_embsize = dec_embsize
        self.map_embsize = map_embsize
        self.race_embsize = race_embsize

        # with striding of 64, that's 16x16 on biggest maps (1024x1024),
        # thus is we reshape the map in a 1D vector that's 8 dims per (x,y)
        self.inp_embsize = self.rnn_input_size // (1024//args.stride)**2
        self.nfeat = args.n_inp_feats * 2
        self.nchannel = self.nfeat + race_embsize * 2 + map_embsize

        # Modules
        self.trunk = _MapRaceFeaturize(args, map_embsize, race_embsize)
        self.conv1x1 = nn.Conv2d(self.nchannel, self.inp_embsize, 1)  # TODO do that before trunk?
        self.rnn = nn.LSTM(self.rnn_input_size, self.hid_dim, self.lstm_num_layers, dropout=self.lstm_dropout)
        self.hidden = self.init_hidden()
        self.decoder = decoder(dec_convsize, dec_convsize)(
            conv        =conv,
            non_lin     =self.nonlin,
            input_size  =self.nchannel + self.hid_dim,
            interm_size =self.dec_embsize,
            output_size =self.dec_embsize,
            depth       =dec_depth
        ) # should be depth=1, the lstm should do work the work.
        self.regression_head = nn.Conv2d(self.dec_embsize, self.nfeat, 1)
        self.unit_class_head = nn.Conv2d(self.dec_embsize, 2 * len(utm.our_units_inds), 1)
        self.bldg_class_head = nn.Conv2d(self.dec_embsize, 2 * len(utm.our_bldgs_inds), 1)
        self.opbt_class_head = nn.Linear(self.dec_embsize, len(utm.nmy_bldgs_inds))
        self.accepts_bptt = True

    def init_hidden(self, bsz=1, rnn=None):
        if rnn is None:
            rnn = self.rnn
        weight = next(rnn.parameters()).data
        return [(Variable(weight.new(self.lstm_num_layers, bsz, self.hid_dim).zero_()),
                Variable(weight.new(self.lstm_num_layers, bsz, self.hid_dim).zero_()))]

    def repackage_hidden(self, hidden):
        if hidden is None:
            return hidden
        elif type(hidden) == Variable:
            return hidden.detach()
        elif type(hidden) == list:
            return [self.repackage_hidden(v) for v in hidden]
        else:
            return tuple(self.repackage_hidden(v) for v in hidden)

    def forward(self, scmap, race, input):
        '''
        scmap: 1xCxHxW features about our game map
        race: 1x2 (my race, their race)
        features: TxFxHxW, with feature dim F and time dim T
        '''
        raise RuntimeError("This is not tested and might not work, check out model 'simple' instead")


class simple(lstm):
    '''
    Simplest model:
        trunk => MapRaceFeaturize concat with input, (F1 x H x W)
        embedding => trunk goes through encoder + spatial-pooling, (F2)
        rnnout => embedding vector goes through RNN, (F3)
        output => rnnout is replicated and concat with trunk, ([F1+F3] x H x W)
        prediction!

    Arguments;
        bypass_encoder: bypass the encoder with a sum pooled version of the input
        enc_embsize: encoder embedding size
        inp_embsize: input embdding size
        top_pooling: how to do the pooling after the encoder, before the RNN can be any of {'mean', 'max', 'sum', 'all'}
        with_z: adds a z latent variable that the encoder needs to predict, concatenated to the LSTM input, whose values are set by backpropagating the loss down to it.
            changes x->enc->lstm->dec->x'
            into x->enc->{z,ex}, {ex,z'}->lstm->dec->x'
            with z'=Parameter, L(z,z')=dist, one z' per game.
    '''
    def __init__(self, args,
                 bypass_encoder=False,
                 enc_convsize=3,
                 enc_embsize=256,
                 enc_depth=3,
                 inp_embsize=256,
                 top_pooling='mean',
                 with_z=False,
                 z_opt=th.optim.SGD,
                 z_lr=0.01,
                 z_lambda=100,
                 z_pred_cut_gradient=False, # Whether to push gradients from the Z loss
                 z_after_lstm=False,  # Whether to do the z model after the LSTM
                 zbwd_init_zfwd=True,  # Whether to init zbwd with zfwd (zpred)
                 zbwd_to_convergence=True,  # Whether to optimize zbwdto convergence of the final loss for each game
                 zbwd_single=False,  # Whether to have only zbwd(game) instead of zbwd(time,game)
                 zfwd_zbwd_ratio=0,  # Ratio of how much of zfwd / (zfwd + zbwd) to put as input to the decoder:
                                     # 0 means only zbwd, 1 means only zfwd, 0.5 means half each.
                 **kwargs):
        kwargs.setdefault('model_name', 'simple')

        self.z_opt = z_opt
        self.enc_embsize = enc_embsize
        self.with_z = with_z
        self.z_after_lstm = z_after_lstm
        rnn_input_size = (enc_embsize * 2 if bypass_encoder else enc_embsize)
        if self.with_z:  # TODO replace by class decorator
            self.zsize = 64
            if not self.z_after_lstm:
                rnn_input_size += self.zsize
            logging.info("rnn input size: {}".format(rnn_input_size))

        super(simple, self).__init__(
            args, lstm_nlayers=1,
            rnn_input_size=rnn_input_size, **kwargs
        )
        assert (self.dec_convsize % 2) == 1, \
            "ERROR: the size of the decoder convolution is not odd"

        self.bypass_encoder = bypass_encoder
        self.append_to_decoder_input = []
        self.predict_delta = args.predict_delta
        self.top_pooling = top_pooling
        self.inp_embsize = inp_embsize

        # Overrides
        self.conv1x1 = nn.Conv2d(self.nchannel, self.inp_embsize, 1)  # TODO do that before trunk?
        if self.residual:
            assert self.inp_embsize == self.enc_embsize, "can't residual from {} to {}".format(self.inp_embsize, self.enc_embsize)
        self.encoder = convnet(
            convsize_0 =3,
            convsize   =5,
            padding_0  =1,
            padding    =2,
            conv       =self.convmod,
            non_lin    =self.nonlin,
            input_size =self.inp_embsize,
            interm_size=self.inp_embsize,
            output_size=self.enc_embsize,
            depth      =2,
            stride_0   =1,
            stride     =2
        )
        self.z_pred_cut_gradient = z_pred_cut_gradient
        if self.with_z:  # TODO replace by class decorator
            self.game_name = None
            zlinear = None
            if self.z_after_lstm:
                zlinear = nn.Linear(self.hid_dim, self.zsize)
            else:
                zlinear = nn.Linear(self.enc_embsize + (0 if not self.bypass_encoder else self.enc_embsize), self.zsize)
            self.zpred = zlinear
            self.zbwd = Variable(th.zeros(1,1,self.zsize).type(th.cuda.FloatTensor))
            self.zbwd.requires_grad = True
            self.zs = {}  # TODO replace by LookUpTable
            self.zlossfn = nn.MSELoss(size_average=True)
            # ^ could also change loss, and make sure the z_lr is small enough!
            self.z_lr = z_lr
            self.z_lambda = z_lambda
            self.zbwd_init_zfwd = zbwd_init_zfwd
            self.zbwd_to_convergence = zbwd_to_convergence
            self.zbwd_single = zbwd_single
            if self.zbwd_single:
                assert self.zbwd_init_zfwd
                assert self.zbwd_to_convergence
            self.zfwd_zbwd_ratio = zfwd_zbwd_ratio
            if self.zfwd_zbwd_ratio > 0:
                assert self.zbwd_init_zfwd

        # TODO decoder that starts from input embedding (after first 1x1 Conv2d)
        # TODO check input/output size in features/channels
        # TODO try to remove border artifacts (borders are important!)
        # TODO hierarchical deconv
        self.decoder = decoder(self.dec_convsize, self.dec_convsize)(
            conv        =self.convmod,
            non_lin     =self.nonlin,
            input_size  =self.nchannel + self.hid_dim + (self.zsize if self.z_after_lstm else 0),
            interm_size =self.dec_embsize,
            output_size =self.dec_embsize,
            depth       =self.dec_depth,
        )

        # Modules
        if self.bypass_encoder:
            self.sum_pool_embed = nn.Linear(self.nfeat, self.enc_embsize)
        if self.top_pooling == 'all':
            self.weight_poolings = nn.Linear(self.enc_embsize, self.enc_embsize * 2)

    def state_dict(self, *args, **kwargs):
        state = super(simple, self).state_dict(*args, **kwargs)
        if self.with_z:
            state['z_dictionary'] = self.zs
        return state

    def load_state_dict(self, state, *args, **kwargs):
        if self.with_z:
            self.zs = state.pop('z_dictionary')
        save = self.state_dict
        __old_state_dict = super(simple, self).state_dict
        self.state_dict = __old_state_dict
        state = super(simple, self).load_state_dict(state, *args, **kwargs)
        self.state_dict = save

    @staticmethod
    def _pool(fn, x):
        # Don't squeeze out batch dim in case of single batch dim
        return fn(fn(x, dim=3), dim=2)

    def init_z(self, game_name):
        if self.with_z:
            if self.game_name is not None: # previous game, we save the backpropagated value of z
                self.zs[self.game_name] = self.zbwd.data.clone()
            self.game_name = game_name
            if game_name is not None and game_name not in self.zs:
                self.zs[self.game_name] = th.zeros(1,1,self.zsize).type(th.cuda.FloatTensor)  # this is gonna be using GPU memory but should be OK (n_games*2*zsize*4 bytes, 71MB with 64dim z)
            self.zbwd_initialized = False

    def pooling(self, x, method=None):
        if method is None:
            method = self.top_pooling
        # pool over map. output bsz x self.enc_embsize
        if method == 'mean':
            x = self._pool(th.mean, x)
        elif method == 'max':
            x = self._pool(lambda x, dim: th.max(x, dim)[0], x)
        elif method == 'sum':
            x = self._pool(th.sum, x)
        elif method == 'all':
            x = th.cat([self._pool(th.sum, x),
                        #self._pool(th.max, x),
                        self._pool(th.mean, x)
                        ], dim=1)
            x = self.weight_poolings(x)
        else:
            logging.error("no pooling if encoding is not exactly 1x1!")
            sys.exit(-1)
        return x  # x is dimensions (length_sequence, self.enc_embsize)

    def do_rnn(self, x, size, hidden):
        bsz = size[0]
        H = size[2]
        W = size[3]
        output, hidden = self.rnn(x, hidden)
        featsize = self.hid_dim
        if self.z_after_lstm and self.with_z:
            output = self.forward_up_to_zfwd_init_zbwd(output)
            featsize += self.zsize
            shaped = self.zfwd.transpose(1, 2).unsqueeze(3).expand(bsz, self.zsize, H, W)
        output = output.transpose(1, 2).unsqueeze(3).expand(bsz, featsize, H, W)
        return output

    def encode(self, x):
        return self.encoder(x)

    def trunk_encode_pool(self, scmap, race, input):
        '''
        also used in subclasses
        scmap: 1xCxHxW features about our game map
        race: 1x2 (my race, their race)
        features: TxFxHxW, with feature dim F and time dim T
        '''
        self.input_sz = input.size()
        if self.bypass_encoder:
            bypass = self.sum_pool_embed(self._pool(th.sum, input))
            bypass = bypass.unsqueeze(1)
        input = self.trunk(scmap, race, input).contiguous() # required for cudnn
        x = self.conv1x1(input)
        x = self.encode(x)
        x = self.pooling(x).unsqueeze(1)
        if self.bypass_encoder:
            x = th.cat([x, bypass], dim=2)
            # bypass has self.enc_embsize for dimension[-1]
            # x has self.enc_embsize for dimension[-1]
        return input, x

    def do_heads(self, x):
        reg = self.regression_head(x)
        uni = self.unit_class_head(x)
        bui = self.bldg_class_head(x)

        opbt = self.opbt_class_head(self.pooling(x, 'max'))
        if not self.predict_delta:
            reg = F.relu(reg, inplace=True)
        return reg, uni, bui, opbt

    def forward_up_to_zfwd_init_zbwd(self, embed):
        if self.z_pred_cut_gradient:
            embed = embed.detach()
        self.zfwd = self.zpred(embed)
        if self.game_name is None: # inference mode
            z = self.zfwd.detach()  # Not necessary but will error if something is wrong
        elif self.zbwd_init_zfwd:
            z = self.zbwd
            if not self.zbwd_initialized:
                if self.zbwd_single:
                    z.data.copy_(self.zfwd.data.mean(0, keepdim=True))
                else:
                    z.data.copy_(self.zfwd.data)
            if self.zbwd_single:
                z = z.expand(embed.size(0), 1, self.zsize)
        else:
            self.zbwd.data.copy_(self.zs[self.game_name])
            z = self.zbwd.repeat(embed.size(0), 1, 1)
        self.zfwd = F.tanh(self.zfwd)
        self.zbwd_initialized = True
        return th.cat([embed, F.tanh(z)], dim=2)

    def forward_rest(self, input, embed):
        self.input, self.embed = input, embed
        if not self.z_after_lstm and self.with_z:
            self.embed = self.forward_up_to_zfwd_init_zbwd(self.embed)

        self.rnn_output = self.do_rnn(self.embed, self.input_sz, self.hidden[-1])

        decoder_input = th.cat([self.input, self.rnn_output] + self.append_to_decoder_input, dim=1)
        return self.do_heads(self.decoder(decoder_input))

    def forward(self, scmap, race, input):
        input, embed = self.trunk_encode_pool(scmap, race, input)
        return self.forward_rest(input, embed)


class striding(simple):
    '''
    Just simple with different parameters:
    lots of downsampling in encoder so we don't have to pool at the very end
    '''
    def __init__(self,
                 args,
                 enc_convsize=5,
                 enc_stride=2,
                 enc_depth=4,
                 **kwargs):
        kwargs.setdefault('model_name', 'striding')
        super(striding, self).__init__(args, **kwargs)
        enc_padding = (enc_convsize - 1) // 2

        # Overrides
        self.encoder = simple_convnet(
            convsize   =enc_convsize,
            padding    =enc_padding,
            conv       =self.convmod,
            non_lin    =self.nonlin,
            input_size =self.inp_embsize,
            output_size=self.enc_embsize,
            depth      =enc_depth,
            stride     =enc_stride
        )


class sum_decode(striding):
    ''' This might not work yet / hasn't been debugged properly, but it runs'''
    def __init__(self, args, **kwargs):
        kwargs.setdefault('model_name', 'sum_decode')
        kwargs['conv'] = _transpose_convtype(kwargs.get('conv', nn.Conv2d))
        super(sum_decode, self).__init__(args, **kwargs)

        # Modules
        self.dec_t = decoder(self.dec_convsize, self.dec_convsize)(
            conv        =self.convmod,
            non_lin     =self.nonlin,
            input_size  =self.nchannel + self.hid_dim,
            interm_size =self.dec_embsize,
            output_size =self.dec_embsize,
            depth       =self.dec_depth,
        )

    def forward(self, scmap, race, input):
        raise RuntimeError("This totally doesn't really work")
        sz = input.size()
        input, embed = self.trunk_encode_pool(scmap, race, input) # in class simple
        output = self.do_rnn(embed, sz, self.hidden[-1])
        skip_connected = th.cat([input, output], dim=1)
        # The above is the same as striding / simple

        move_in = self.decoder(skip_connected)  # biases are creation/destr
        move_out = self.dec_t(skip_connected)  # biases are destruction/crea
        ret = move_in + move_out

        return self.do_heads(ret) # in class simple


class multilvl_lstm(simple):
    '''
    Structurally the same as simple, but has LSTMs not just on the top layer.
    Therefore, this only does a 2x2 pool per n_lvl, so the defaults does a
    4x4 pooling total.
    Additional to replicating the top level LSTM, the mid-level LSTMs are
    replicated into the right bin with an upsample_bilinear (or nearest?)
    '''
    def __init__(self, args,
                 midconv_kw=3,
                 midconv_stride=2,
                 midconv_depth=2,
                 n_lvls=2,
                 upsample='bilinear',
                 use_mid_rnns_in_encoding=True,
                 **kwargs):
        assert n_lvls > 0, "n_lvls must be at least 1"
        self.n_lvls = n_lvls
        kwargs.setdefault('model_name', 'multilvl_lstm')
        super(multilvl_lstm, self).__init__(args, **kwargs)
        midconv_padding = (midconv_kw - 1) // 2

        # Overrides
        self.encoder = None  # TODO I hope this garbage collects.
        # TODO actually do it with the nn.Module correspondong to midnets+midrnn
        # TODO skip connections from midnets to corresponding decoder level.

        self.decoder = decoder(self.dec_convsize, self.dec_convsize)(
            conv        =self.convmod,
            non_lin     =self.nonlin,
            input_size  =self.nchannel + self.hid_dim + self.enc_embsize * self.n_lvls + (self.zsize if self.z_after_lstm else 0),
            interm_size =self.dec_embsize,
            output_size =self.dec_embsize,
            depth       =self.dec_depth,
        ) # should be depth=1, the lstm should do work the work.
        self.rnn_input_size = self.inp_embsize + (0 if not self.bypass_encoder else self.enc_embsize) + (0 if not self.with_z or self.z_after_lstm else self.zsize)
        self.rnn = nn.LSTM(self.rnn_input_size, self.hid_dim, self.lstm_num_layers, dropout=self.lstm_dropout)

        # Modules
        self.use_mid_rnns_in_encoding = use_mid_rnns_in_encoding
        self.midnets = nn.ModuleList()
        self.midrnn = nn.ModuleList()
        for i in range(n_lvls):
            isize = self.enc_embsize
            osize = self.enc_embsize
            if i == 0:
                isize = self.inp_embsize
            self.midnets.append(nn.Sequential(
                simple_convnet(
                    convsize   =midconv_kw,
                    padding    =midconv_padding,
                    conv       =self.convmod,
                    non_lin    =self.nonlin,
                    input_size =isize,
                    output_size=osize,
                    depth      =midconv_depth - 1,
                    stride     =1
                ),
                self.convmod(isize, osize, 3, 2, padding=1)
            ))
            self.midrnn.append(nn.LSTM(isize, osize, 1, dropout=self.lstm_dropout))

        self.upsample = {
            'bilinear': F.upsample_bilinear,
            'nearest': F.upsample_nearest,
        }[upsample]

    def encode(self, x):
        # Each midnet runs a convolution + LSTM; TODO make it do multiple conv
        self.append_to_decoder_input = []
        for i, conv in enumerate(self.midnets):
            x = self.nonlin(conv(x))
            if self.use_mid_rnns_in_encoding:
                x = self.do_rnn_middle(x, self.input_sz, i)
                self.append_to_decoder_input.append(x)
            else:
                self.append_to_decoder_input.append(self.do_rnn_middle(x, self.input_sz, i))
        return x

    def init_hidden(self):
        return [None for _ in range(self.n_lvls)] + super(multilvl_lstm, self).init_hidden()

    def do_rnn_middle(self, x, sz, i):
        xs2 = x.size(2)
        xs3 = x.size(3)
        if self.hidden[i] == None:
            n_blocks = xs2 * xs3
            self.hidden[i] = super(multilvl_lstm, self).init_hidden(n_blocks, self.midrnn[i])[0]
        bsz = sz[0]
        H = sz[2]
        W = sz[3]
        x = x.view(sz[0], self.enc_embsize, -1).transpose(1, 2)
        output, self.hidden[i] = self.midrnn[i](x, self.hidden[i])
        output = output.transpose(1, 2).contiguous().view(bsz, self.enc_embsize, xs2, xs3)
        output = self.upsample(output, size=(H, W))
        return output


class conv_only(striding):
    '''
    This does the typical encoding, but does a conv1d instead of an rnn
    at the end
    '''
    def __init__(self, args, **kwargs):
        kwargs.setdefault('model_name', 'conv_only')
        super(conv_only, self).__init__(args, **kwargs)
        self.accepts_bptt = False

        # Modules
        self.temporal_convmod = residual_layer(nn.Conv1d, self.nonlin) if self.residual else nn.Conv1d
        self.temporal_conv = convnet(
            convsize_0 =3,
            convsize   =3,
            padding    =1,
            padding_0  =1,
            conv       =self.temporal_convmod,
            non_lin    =self.nonlin,
            input_size =self.enc_embsize + (0 if not self.with_z else self.zsize) + (0 if not self.bypass_encoder else self.enc_embsize),
            interm_size=self.hid_dim,
            output_size=self.hid_dim,
            depth      =3,
            stride_0   =1,
            stride     =1,
        )
        self.pad = 2
        # TODO Automaticall calculate padding
        # TODO maybe do some pooling in the temporal convolution

    def temp_conv(self, embed):
        bsz = embed.size(0)
        if self.with_z:
            if self.game_name is not None and self.game_name in self.zs:  # train
                embed = th.cat([embed, self.zbwd.expand((bsz, 1,
                    self.zsize))], dim=2)
            else:  # valid / test
                zbwd = self.zbwd.expand((bsz, 1, self.zsize))
                zbwd.data.copy_(self.zfwd.data)
                embed = th.cat([embed, zbwd], dim=2)
        with th.cuda.device_of(embed.data): # can't use nn.pad so we do this
            pad = Variable(embed.data.new().resize_(self.pad, 1, embed.size(2)).zero_(), requires_grad=False)
            embed = th.cat((pad,embed), dim=0)
            embed = embed.transpose(0, 1).transpose(1, 2).contiguous()
        return self.temporal_conv(embed)

    def forward(self, scmap, race, input):
        sz = input.size()
        input, embed = self.trunk_encode_pool(scmap, race, input)
        raise RuntimeError("Need to change this to have the option to do "
                           "Z after or before the TemporalConv")
        if self.with_z:
            self.zfwd = self.zpred(embed)
            if self.game_name is not None and self.game_name in self.zs:
                zvalue = self.zs[self.game_name]
                zvar = Variable(zvalue).expand((input.size(0), 1, self.zsize))
                loss = self.zlossfn(self.zfwd, zvar)
                loss.backward(retain_graph=True)
        output = self.temp_conv(embed)
        output = output.narrow(2, 0, sz[0])
        output = output.transpose(0, 2).unsqueeze(3).expand(sz[0], self.hid_dim, sz[2], sz[3])
        skip_connected = th.cat([input, output], dim=1)

        return self.do_heads(self.decoder(skip_connected))

