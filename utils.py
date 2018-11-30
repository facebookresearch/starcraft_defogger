# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchcraft import Constants
import _ext
import torch as th

import os, time, subprocess
import numpy as np
import logging
from itertools import chain
from collections import defaultdict


datadir = "/datasets01/starcraft_replays-reduced/060817"
def setup(parser, fn_args=None):
    # NOTE: If you change the defaults, you better go change mark what changed
    # in the results in the README.md too
    parser.add_argument('-i', '--input', default=datadir,
                        help="Input directory for input files. Must have a "
                        "${prefix}train.list, ${prefix}valid.list, and "
                        "${prefix}test.list files")
    parser.add_argument('--load', default=False,
                        help="Whether to load the model")
    parser.add_argument('--finetune', type=str, default="",
                        help="Whether to load a model if it doesn't exist")
    parser.add_argument('--save', default=False,
                        help="Whether to save the model")
    parser.add_argument('--io', default=False,
                        help="Shorthand for --load and --save to the same file")
    parser.add_argument('--small', default=False, action='store_true',
                        help="Whether to use smalltrain.list, ... instead")
    parser.add_argument('--check_dims', default=False, action='store_true',
                        help="Whether to show all dims (features, model)")
    parser.add_argument('--data_threads', default=10, type=int,
                        help="Number of threads to use for dataloader")
    parser.add_argument('--skip_frames', default=80, type=int,
                        help="Number of skip frames (multiply by 3 to get true #)")
    parser.add_argument('--combine_frames', default=80, type=int,
                        help="Number of combine frames. (multiply by 3)")
    parser.add_argument('--debug', default=3, type=int,
                        help="Choices for debug, can be 1-5, with 1 most verbose")
    parser.add_argument('-k', '--kernel_size', default=64, type=int,
                        help="featurizer kernel size")
    parser.add_argument('-s', '--stride', default=0, type=int,
                        help="featurizer stride size, secretly defaults to kernel size")
    parser.add_argument('--divide_by', default=100, type=int,
                        help="divide features by this number; special values: 0 -> normalize features by their sum space, per timestep; negative -> do nothing")
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help="which GPU to use, -1 for CPU")
    parser.add_argument('--reduced', default=False, action='store_true',
                        help='data is in featurizer-specific reduced format')
    parser.add_argument('--predict', default='defog',
                        help="what prediction mode to use, between: {'defog', 'only_us', 'full_info', 'only_defog'}")
    parser.add_argument('--predict_delta', default=False, action='store_true',
                        help="predict the delta between t and t+1")
    parser.add_argument('--class_prob_thresh', default=0.5, type=float,
                        help="Existence when p[building] is greater than this flag")
    parser.add_argument('--until_time', default=0, type=int,
                        help="predict until time (in minutes), 0 for full game")
    parser.add_argument('--from_time', default=0, type=int,
                        help="predict from time (in minutes), 0 for full game")
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help="# epochs")
    parser.add_argument('--check_nan', default=False, action='store_true',
                        help="Check nan values during training, very very slow")
    parser.add_argument('--just_valid', default=False, action='store_true',
                        help="Just do a validation run")
    parser.add_argument('--use_true_visibility', default=False, action='store_true',
                        help="Use Jonas's visibility map, which is strictly more visible than reality. Default is strictly less visible")
    parser.add_argument('--n_unit_thresh', default=0.1, type=float,
                        help="If model predicts > n_unit_thresh, round it to the next whole number")
    parser.add_argument('--regr_slope_scalar', default=1.0, type=float,
                        help="If model predict K units, return K*regr_slope_scalar units")

    args = parser.parse_args() if fn_args is None else parser.parse_args(fn_args)
    if ((args.save is not False or args.load is not False)
            and args.io is not False):
        raise RuntimeError("Cannot have save or load with --io option")
    if args.io is not False:
        args.save = args.io
        args.load = args.io
    if args.save is not False:
        args.save = os.path.abspath(args.save)
        os.makedirs(args.save, exist_ok=True)
    if args.check_dims:
        args.debug = min(args.debug, 2)

    logFormatter = logging.Formatter("[%(asctime)s %(filename)s:%(lineno)-4d] | %(levelname)8s | %(message)s", "%m-%d %H:%M:%S")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(0)

    if args.save is not False:
        fileHandler = logging.FileHandler(os.path.join(args.save, "defogger.log"))
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(40)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(args.debug*10)
    rootLogger.addHandler(consoleHandler)

    if args.debug <= 1:
        th.backends.cudnn.enabled = False
    else:
        th.backends.cudnn.enabled = True

    # TODO skip_frames >= combine_frames
    assert args.skip_frames >= args.combine_frames, ("Not yet supported, need "
        "to implement predict forward more than 1 skip frame.")
    args.stride = args.kernel_size if args.stride == 0 else args.stride

    args.from_time = convert_minutes_to_repframes(args.from_time, args.combine_frames)
    args.until_time = convert_minutes_to_repframes(args.until_time, args.combine_frames)

    if 'SLURM_JOBID' in os.environ:
        logging.log(42, "slurm_info {} {} {}".format(
            os.environ.get('SLURM_JOBID', -1),
            os.environ.get('SLURM_ARRAY_JOB_ID', -1),
            os.environ.get('SLURM_ARRAY_TASK_ID', -1),
        ))
    logging.log(42, "git_commit {}".format(subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()))
    logging.log(42, "process_pid {}".format(os.getpid()))
    for a, b in sorted(vars(args).items()):
        logging.log(42, "argument {} {}".format(a, b))

    featurizer = _ext.CoarseConvFeaturizerUnitTypes(
        args.kernel_size, args.kernel_size, args.stride, args.stride,
        args.from_time, args.until_time)
    args.featurizer = featurizer
    args.n_inp_feats = args.featurizer.feature_size
    # Finds the breakoff for units and buildings
    ind2unit = {Constants.unittypes._dict[i] : v for i, v in
                   enumerate(featurizer.typemapper) if v != featurizer.feature_size - 1}
    args.unit_range = (0, ind2unit['Zerg_Lurker'] + 1)
    args.bldg_range = (ind2unit['Terran_Command_Center'], ind2unit['Protoss_Shield_Battery'] + 1)
    # everything after cutoff is a building inclusive, except for the last,
    # which is unknown
    return args


def normalize_by_first_dim(nparr):
    norm = nparr.sum(axis=tuple(range(1, nparr.ndim))) + 1e-10
    return nparr / norm[tuple(
        chain([slice(None)], [np.newaxis for x in range(nparr.ndim-1)]))]


def divide_by(number):
    def ret(nparr):
        return nparr / number
    return ret


def hash_map(map):
    ''' We assume all maps with the same hash are equal... '''
    size = map['walkability'].shape
    sum_walk = map['walkability'].sum()
    sum_height = map['ground_height'].sum()
    return tuple(chain(size, [sum_walk, sum_height]))


def get_info(rep):
    ''' Returns the race and start location. '''
    # We use frame 24 because some maps have observe triggers and units don't
    # spawn on frame 0
    p0 = {x.type: x for x in rep.getFrame(3).units[0]}
    p1 = {x.type: x for x in rep.getFrame(3).units[1]}
    mapper = {
        Constants.unittypes.Protoss_Nexus: 0,
        Constants.unittypes.Zerg_Hatchery: 1,
        Constants.unittypes.Terran_Command_Center: 2,
    }
    for k, v in mapper.items():
        if k in p0:
            s0x, s0y = p0[k].x, p0[k].y
            p0 = v
            break
    else:
        raise RuntimeError("Can't find a race for player 0")
    for k, v in mapper.items():
        if k in p1:
            s1x, s1y = p1[k].x, p1[k].y
            p1 = v
            break
    else:
        raise RuntimeError("Can't find a race for player 0")
    return [(p0, s0x, s0y), (p1, s1x, s1y)]


def zero_init(model):
    for param in model.parameters():
        param.data.zero_()
    return model

def div_init(model, n):
    for param in model.parameters():
        param.data.div_(n)
    return model

def make_start_loc(locations, shape):
    ''' Gives a 1-hot matrixwhere the start locations are given a shape '''
    sl = np.zeros(shape)
    for x, y in locations:
        sl[y][x] = 1
    return sl

def get_normalize_fn(args):
    if args.divide_by > 0:
        return lambda x: x / args.divide_by
    elif args.divide_by == 0:
        return normalize_by_first_dim
    else:
        return lambda x: x


def convert_minutes_to_repframes(m, combine_frames):
    # (3) real frames / feature frame
    # 42 ms / real frame
    return int(m / (3.0) / 0.042 * 60)


class State(object):
    def __init__(self, items):
        self.__dict__.update(items)

    def __setattr__(self, name, value):
        if name not in self.__dict__: raise RuntimeError("{} is not a state member".format(name))
        object.__setattr__(self, name, value)

    def __repr__(self):
        return repr(self.__dict__)


class __UnitTypesMapper(object):
    def __init__(self):
        featurizer = _ext.CoarseConvFeaturizerUnitTypes()
        self.tc_to_feats = featurizer.typemapper

        self.feats_to_tc = [self.tc_to_feats.index(i) for i in range(117)] + [None]
        self.offset = featurizer.feature_size

        self.our_bldgs_inds = [e for e in filter(lambda x: x<117, self.tc_to_feats[106:])]
        self.our_units_inds = [e for e in filter(lambda x: x<117, self.tc_to_feats[:106])]
        self.nmy_bldgs_inds = [e + self.offset for e in self.our_bldgs_inds]
        self.nmy_units_inds = [e + self.offset for e in self.our_units_inds]
        cut = Constants.unittypes
        req = defaultdict(dict)
        req[cut.Terran_Marine][cut.Terran_Barracks] = 1
        req[cut.Terran_Ghost][cut.Terran_Academy] = 1
        req[cut.Terran_Ghost][cut.Terran_Covert_Ops] = 1
        req[cut.Terran_Goliath][cut.Terran_Armory] = 1
        req[cut.Terran_Machine_Shop][cut.Terran_Factory] = 1
        req[cut.Terran_Siege_Tank_Tank_Mode][cut.Terran_Machine_Shop] = 1
        req[cut.Terran_Vulture][cut.Terran_Factory] = 1
        req[cut.Terran_Wraith][cut.Terran_Starport] = 1
        req[cut.Terran_Control_Tower][cut.Terran_Starport] = 1
        req[cut.Terran_Science_Vessel][cut.Terran_Control_Tower] = 1
        req[cut.Terran_Science_Vessel][cut.Terran_Science_Facility] = 1
        req[cut.Terran_Dropship][cut.Terran_Control_Tower] = 1
        req[cut.Terran_Battlecruiser][cut.Terran_Control_Tower] = 1
        req[cut.Terran_Battlecruiser][cut.Terran_Physics_Lab] = 1
        req[cut.Terran_Physics_Lab][cut.Terran_Science_Facility] = 1
        req[cut.Terran_Covert_Ops][cut.Terran_Science_Facility] = 1
        req[cut.Terran_Siege_Tank_Siege_Mode][cut.Terran_Machine_Shop] = 1
        req[cut.Terran_Firebat][cut.Terran_Academy] = 1
        req[cut.Terran_Medic][cut.Terran_Academy] = 1
        req[cut.Zerg_Zergling][cut.Zerg_Spawning_Pool] = 1
        req[cut.Zerg_Hydralisk][cut.Zerg_Hydralisk_Den] = 1
        req[cut.Zerg_Lurker][cut.Zerg_Hydralisk_Den] = 1
        req[cut.Zerg_Lurker_Egg][cut.Zerg_Hydralisk_Den] = 1
        req[cut.Zerg_Ultralisk][cut.Zerg_Ultralisk_Cavern] = 1
        req[cut.Zerg_Mutalisk][cut.Zerg_Spire] = 1
        req[cut.Zerg_Guardian][cut.Zerg_Greater_Spire] = 1
        req[cut.Zerg_Queen][cut.Zerg_Queens_Nest] = 1
        req[cut.Zerg_Defiler][cut.Zerg_Defiler_Mound] = 1
        req[cut.Zerg_Scourge][cut.Zerg_Spire] = 1
        req[cut.Terran_Valkyrie][cut.Terran_Control_Tower] = 1
        req[cut.Terran_Valkyrie][cut.Terran_Armory] = 1
        req[cut.Zerg_Cocoon][cut.Zerg_Greater_Spire] = 1
        req[cut.Protoss_Dark_Templar][cut.Protoss_Templar_Archives] = 1
        req[cut.Zerg_Devourer][cut.Zerg_Greater_Spire] = 1
        req[cut.Protoss_Zealot][cut.Protoss_Gateway] = 1
        req[cut.Protoss_Dragoon][cut.Protoss_Cybernetics_Core] = 1
        req[cut.Protoss_High_Templar][cut.Protoss_Templar_Archives] = 1
        req[cut.Protoss_Archon][cut.Protoss_Templar_Archives] = 1
        req[cut.Protoss_Dark_Archon][cut.Protoss_Templar_Archives] = 1
        req[cut.Protoss_Arbiter][cut.Protoss_Arbiter_Tribunal] = 1
        req[cut.Protoss_Carrier][cut.Protoss_Fleet_Beacon] = 1
        req[cut.Protoss_Reaver][cut.Protoss_Robotics_Support_Bay] = 1
        req[cut.Protoss_Observer][cut.Protoss_Observatory] = 1
        req[cut.Protoss_Shuttle][cut.Protoss_Robotics_Facility] = 1
        req[cut.Protoss_Scout][cut.Protoss_Stargate] = 1
        req[cut.Protoss_Corsair][cut.Protoss_Stargate] = 1
        req[cut.Terran_Comsat_Station][cut.Terran_Academy] = 1
        req[cut.Terran_Nuclear_Silo][cut.Terran_Science_Facility] = 1
        req[cut.Terran_Nuclear_Silo][cut.Terran_Covert_Ops] = 1
        req[cut.Terran_Barracks][cut.Terran_Command_Center] = 1
        req[cut.Terran_Academy][cut.Terran_Barracks] = 1
        req[cut.Terran_Factory][cut.Terran_Barracks] = 1
        req[cut.Terran_Starport][cut.Terran_Factory] = 1
        req[cut.Terran_Science_Facility][cut.Terran_Starport] = 1
        req[cut.Terran_Engineering_Bay][cut.Terran_Command_Center] = 1
        req[cut.Terran_Armory][cut.Terran_Factory] = 1
        req[cut.Terran_Missile_Turret][cut.Terran_Engineering_Bay] = 1
        req[cut.Terran_Bunker][cut.Terran_Barracks] = 1
        req[cut.Zerg_Lair][cut.Zerg_Spawning_Pool] = 1
        req[cut.Zerg_Hive][cut.Zerg_Queens_Nest] = 1
        req[cut.Zerg_Nydus_Canal][cut.Zerg_Hive] = 1
        req[cut.Zerg_Hydralisk_Den][cut.Zerg_Spawning_Pool] = 1
        req[cut.Zerg_Defiler_Mound][cut.Zerg_Hive] = 1
        req[cut.Zerg_Queens_Nest][cut.Zerg_Lair] = 1
        req[cut.Zerg_Evolution_Chamber][cut.Zerg_Hatchery] = 1
        req[cut.Zerg_Ultralisk_Cavern][cut.Zerg_Hive] = 1
        req[cut.Zerg_Spire][cut.Zerg_Lair] = 1
        req[cut.Zerg_Greater_Spire][cut.Zerg_Hive] = 1
        req[cut.Zerg_Spawning_Pool][cut.Zerg_Hatchery] = 1
        req[cut.Zerg_Spore_Colony][cut.Zerg_Evolution_Chamber] = 1
        req[cut.Zerg_Sunken_Colony][cut.Zerg_Spawning_Pool] = 1
        req[cut.Protoss_Robotics_Facility][cut.Protoss_Cybernetics_Core] = 1
        req[cut.Protoss_Observatory][cut.Protoss_Robotics_Facility] = 1
        req[cut.Protoss_Gateway][cut.Protoss_Nexus] = 1
        req[cut.Protoss_Photon_Cannon][cut.Protoss_Forge] = 1
        req[cut.Protoss_Citadel_of_Adun][cut.Protoss_Cybernetics_Core] = 1
        req[cut.Protoss_Cybernetics_Core][cut.Protoss_Gateway] = 1
        req[cut.Protoss_Templar_Archives][cut.Protoss_Citadel_of_Adun] = 1
        req[cut.Protoss_Forge][cut.Protoss_Nexus] = 1
        req[cut.Protoss_Stargate][cut.Protoss_Cybernetics_Core] = 1
        req[cut.Protoss_Fleet_Beacon][cut.Protoss_Stargate] = 1
        req[cut.Protoss_Arbiter_Tribunal][cut.Protoss_Templar_Archives] = 1
        req[cut.Protoss_Arbiter_Tribunal][cut.Protoss_Stargate] = 1
        req[cut.Protoss_Robotics_Support_Bay][cut.Protoss_Robotics_Facility] = 1
        req[cut.Protoss_Shield_Battery][cut.Protoss_Gateway] = 1
        n_feats_ut = len(self.feats_to_tc)
        self.obs_to_hidden = np.eye(n_feats_ut, dtype='bool')
        for unittype, requirements in req.items():
            for requirement in requirements.keys():
                #print("{}: {} requires {}: {}".format(unittype, cut._dict[unittype], requirement, cut._dict[requirement]))
                #print("{}: {} requires {}: {}".format(self.tc_to_feats[unittype], self.feats_to_string(self.tc_to_feats[unittype]), self.tc_to_feats[requirement], self.feats_to_string(self.tc_to_feats[requirement])))
                self.obs_to_hidden[self.tc_to_feats[unittype]][self.tc_to_feats[requirement]] = True
        for depth in range(8):  # max depth build trees?
            for ut in range(n_feats_ut):
                self.obs_to_hidden[ut] = np.max(self.obs_to_hidden[self.obs_to_hidden[ut]], axis=0)

    def string_to_feats(self, t):
        return self.tc_to_feats[Constants.unittypes[t]]

    def feats_to_string(self, t):
        return Constants.unittypes._dict[self.feats_to_tc[t]]

    def is_building(self, tctype):
        return tctype >= 106  # Terran_Command_Center

    def ttrules(self, tt):
        assert tt.shape[1] == self.obs_to_hidden.shape[1], "tech tree is not of the same dim ({}) as self.obs_to_hidden ({})".format(tt.shape[1], self.obs_to_hidden.shape[1])
        return np.dot(tt, self.obs_to_hidden)

UnitTypesMapper = __UnitTypesMapper()

class Timer:    
    def __init__(self, msg, sync=False):
        self.msg = msg
        self.sync = sync

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        th.cuda.synchronize()
        self.end = time.time()
        self.interval = self.end - self.start
        logging.info("{}: {} s".format(self.msg, self.interval))
