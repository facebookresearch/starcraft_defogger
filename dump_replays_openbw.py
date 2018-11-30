# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchcraft as tc
import torchcraft.Constants as tcc
import argparse
import os
import os.path as path
import sys
import subprocess
import signal
import socket
from contextlib import closing

OPENBW_REPLAY_TEMPLATE = (
    ' TORCHCRAFT_PORT={port}'
    ' OPENBW_MPQ_PATH=/starcraft'
    ' OPENBW_ENABLE_UI=0'
    ' BWAPI_CONFIG_AI__AI="{bwenv}"'
    ' BWAPI_CONFIG_AUTO_MENU__MAP="{map}"'
    ' BWAPI_CONFIG_AUTO_MENU__CHARACTER_NAME="BWEnv"'
    ' BWAPI_CONFIG_AUTO_MENU__AUTO_MENU=SINGLE_PLAYER'
    ' BWAPI_CONFIG_AUTO_MENU__GAME_TYPE=MELEE'
    ' BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART=OFF'
    ' BWAPILauncher'
)

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def dump_replay(path, dest, bwenv):
    print('>> Dumping {} -> {}'.format(path, dest))

    port = find_free_port()
    cmdline = OPENBW_REPLAY_TEMPLATE.format(port=port, bwenv=bwenv, map=path)

    openbw = subprocess.Popen(cmdline, shell=True, preexec_fn=os.setsid)

    cl = tc.Client()
    cl.connect('localhost', port)
    state = cl.init()
    skip_frames = 3
    cl.send([
        [tcc.set_speed, 0],
        [tcc.set_gui, 0],
        [tcc.set_combine_frames, skip_frames, skip_frames],
        [tcc.set_max_frame_time_ms, 0],
        [tcc.set_blocking, 0],
        [tcc.set_frameskip, 1000],
        [tcc.set_log, 0],
        [tcc.set_cmd_optim, 1],
    ])
    state = cl.recv()

    rep = tc.replayer.Replayer()
    rep.setMapFromState(state)
    while not state.game_ended:
        rep.push(state.frame)
        state = cl.recv()

    rep.setKeyFrame(-1)
    rep.save(dest, True)

    # Bye, bye
    os.killpg(os.getpgid(openbw.pid), signal.SIGTERM)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--bwenv', default='TorchCraft/BWEnv/build/BWEnv.so',
                        help='Path to BWEnv')
    parser.add_argument('-o', '--out', default="/tmp", 
                        help="Where to save the replays")
    parser.add_argument('-p','--path-retain',  default=0, type=int,
                        help='number of path levels to retain')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('replays', nargs='+', help='replay files to dump')
    args = parser.parse_args()

    for replay in args.replays:
        replay = path.abspath(replay)
        dest = path.splitext(path.basename(replay))[0] + '.tcr'
        dir = path.dirname(replay)
        for p in range(args.path_retain):
            dest = path.join(path.basename(dir), dest)
            dir = path.dirname(dir)
        dest = path.join(args.out, dest)

        if not args.overwrite and path.exists(dest):
            print('>> Found existing file at {}, skipping'.format(dest))
            continue

        try:
            os.makedirs(path.dirname(dest), exist_ok=True)
            dump_replay(path=replay, dest=dest, bwenv=args.bwenv)
        except Exception as e:
            print('!! Got execption for {}: {}'.format(replay, str(e)))

if __name__ == '__main__':
    main()
