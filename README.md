# Starcraft Defogger

This repository is a code dump of the experiments in
[Forward Modeling for Partial Observation Strategy Games - A StarCraft Defogger](https://papers.nips.cc/paper/8272-forward-modeling-for-partial-observation-strategy-games-a-starcraft-defogger)

## Prerequisites

- PyTorch 0.3.1
- ZSTD 1.3+
- [optional] Visdom
- [Download the dataset](https://s3-us-west-2.amazonaws.com/stardata/original_replays.tar.gz).
  This dataset is built by running `python reduce_data.py --list /path/to/stardata/all.list --save /output/path` on [StarData](https://github.com/TorchCraft/StarData).
  We provide it here since the data is much smaller than StarData after preprocessing.

## Install

    git submodule update --init --recursive
    cd TorchCraft
    pip install .
    cd ..
    make

## Training

Does it work?

    python sweep.py --run test

Train a model

    python sweep.py --run 007_str-kw3_100801010802000000010000 

Test that model on the validation set

    python sweep.py --run 007_str-kw3_100801010802000000010000 --just_valid --class_prob_thresh 0.3 --n_unit_thresh 0.3 --regr_slope_scalar 1.2

Test that model on the test_set

    python sweep.py --run 007_ml_000801010002042201000000 --just_test --class_prob_thresh 0.5 --n_unit_thresh 0.0 --regr_slope_scalar 1

What are the IDs? If you look in `sweep.py`, you'll see a line that looks like:

    generator = serialize(
        k_s_fs,
        losses,
        ...
    )

The numbers are just indices in this list - the first two numbers being 08
means the ID maps to the 8th index in the `k_s_fs` list, which is 128 grid
size, 64 skip size, 120 frame skip, and flags "--predict 'defog'
--predict_delta".

The models used in the paper are:

    007_ml_000801010002024400000000
    007_ml_100801010200022200000000
    007_ml_080801010200000000000000
    007_ml_050801010702000000000000
    007_ml_090801010002000000000000
    007_str-kw3_110801040202000000000000
    007_str-kw3_100601010802000000000000
    007_str-kw3_080601010002000000000000
    007_str-kw3_050601010002000000000000
    007_str-kw3_090601010002000000000000

Our hyperparameter sweeps showed that these were the best in terms of validation
error, and we reported their test results in our table.

## Running Defogger in a StarCraft bot

Once you have a trained model, dump it out with 

    python sweep.py --run $ID --dump_npz

This should put it in `./defogger/$ID/*.npz`. Then, build [TorchCraftAI](https://github.com/torchcraft/TorchCraftAI).
Edit `TorchCraftAI/scripts/defogger/play-with-defogger.cpp` to create a defogger model with the same parameters as the one you trained.
Next, download a BWAPI 4.2.0 bot dll, and name it `420_{race}_{any_name}.dll`.
Then, you can play a game from TorchCraftAI with

    ./build/scripts/defogger/play-with-defogger -map "maps/(4)Fighting Spirit.scx" -opponent 420_Z_ZZZK.dll -blocking -model_path /path/to/your//multilvl_lstm.npz

To use a bot from BWAPI 3.74 or 4.12, the -opponent field expects a name in the
format of `{BWAPI_version}_{race}_{name}.dll`, as well as for you to have
installed [bwapi-bot-loader](https://github.com/tscmoo/bwapi-bot-loader) so
that you can use previous BWAPI versions.

We also provide the dumped replays from the [AIIDE 2017 Starcraft Bot contest](https://www.cs.mun.ca/~dchurchill/starcraftaicomp/2017/),
since we hypothesize that training on this data will get better results for bot-vs-bot games.
We provide the reduced replays [here](https://s3-us-west-2.amazonaws.com/stardata/defogger_reduced_aiide17.tar.gz).

We thank [Florentin Guth](https://github.com/Kegnarok) for the C++ port of the defogger.

## Troubleshooting

"I sometimes see:"

    [03-27 16:15:17 data.py:117 ] |  WARNING | Try 0: Exception while featurizing replay /path/to/replay.tcr.npz.

    Traceback (most recent call last):
      File "starcraft_defogger/data.py", line 112, in _consumer
        res = featurizer(fn)
      File "starcraft_defogger/conv_lstm_utils.py", line 34, in featurize_reduced
        input = feats[0]
    IndexError: list index out of range

    [03-27 16:15:17 data.py:121 ] |  WARNING | Tried to featurize replay /path/to/replay.tcr.npz 1 times,giving up

This error is safe to ignore.


## Citation

Please cite the [NIPS paper](https://papers.nips.cc/paper/8272-forward-modeling-for-partial-observation-strategy-games-a-starcraft-defogger) if you use the defogger in your work
```
@incollection{NIPS2018_8272,
title = {Forward Modeling for Partial Observation Strategy Games - A StarCraft Defogger},
author = {Synnaeve, Gabriel and Lin, Zeming and Gehring, Jonas and Gant, Dan and Mella, Vegard and Khalidov, Vasil and Carion, Nicolas and Usunier, Nicolas},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {10759--10770},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8272-forward-modeling-for-partial-observation-strategy-games-a-starcraft-defogger.pdf}
}
```
