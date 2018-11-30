# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import queue
from os import path
import threading
import random
import traceback
import numpy
import torch
from time import sleep
from itertools import chain


# TODO Make the data loaders all generators
class DataLoader(object):
    def __init__(self, args, featurizer, postfn=lambda x: x):
        self.args = args
        self.nthreads = args.data_threads
        self.featurizer = featurizer
        self.postprocess = postfn
        if args.check_dims:
            def verbose_featurizer(x):
                feats = featurizer(x)
                def get_shape(x):
                    #if type(x).__module__ == 'numpy':
                    if type(x) == numpy.ndarray:
                        return x.shape
                    #elif x.__module__ == 'torch.autograd.variable':
                    elif isinstance(x, torch.autograd.Variable):
                        return x.data.size()
                    else:
                        return None
                if hasattr(feats, '__iter__') and not type(x).__module__ == 'numpy':
                    shapes = [get_shape(x) for x in feats]
                else:
                    shapes = get_shape(x)
                logging.info("Features: {}".format(shapes))
                return feats
            self.featurizer = verbose_featurizer
        prefix = path.join(args.input, "small" if args.small else "")
        self.trfn = prefix + "train.list"
        self.vafn = prefix + "valid.list"
        self.tefn = prefix + "test.list"
        self._done = threading.Event()
        self._done.set()
        self._feeder_queue = queue.Queue()  # TODO check
        self._data_queue = queue.Queue(self.nthreads)
        self._ndone = 0

        self._workers = []
        self._uid = 0;

    def train(self, **kwargs):
        # TODO? Make dl.train(); dl.train() seamless (not stopping)
        self.stop()
        self._load(self.trfn, **kwargs)

    def valid(self, **kwargs):
        self.stop()
        self._load(self.vafn, **kwargs)

    def test(self, **kwargs):
        self.stop()
        self._load(self.tefn, **kwargs)

    def all(self, **kwargs):
        self.stop()
        self._load(self.trfn, self.vafn, self.tefn, **kwargs)

    def file(self, fn, **kwargs):
        self.stop()
        self._load(fn, **kwargs)

    def list(self, data, **kwargs):
        self.stop()
        self._load(data, **kwargs)

    def one(self, fn, **kwargs):
        fn = path.join(self.args.input, fn)
        for i in range(1):  # 1 retries
            try:
                res = self.featurizer(fn)
                break
            except Exception:
                logging.warning("Try {}: "
                        "Exception while featurizing replay {}.\n\n{}"
                        .format(i, fn, traceback.format_exc()))
                sleep(0.05)
        else:
            logging.warning("Tried to featurize replay {} 3 times,"
                    "giving up".format(fn))
            return None
        return self.postprocess(res)
        

    def _consumer(input_q, data_q, done, featurizer):
        tn = threading.current_thread().name
        while True:
            if done.is_set():
                break
            try:
                fn = input_q.get(timeout=1)
            except queue.Empty:
                break
            except Exception:
                logging.info("Exception while getting the next replay.\n\n{}"
                        .format(input_q, traceback.format_exc()))
            bn = "/".join(path.abspath(fn).split('/')[-2:])
            logging.debug("Thread {} is loading replay {}".format(tn, bn))
            for i in range(1):  # 1 retries
                try:
                    res = featurizer(fn)
                    break
                except Exception:
                    logging.warning("Try {}: "
                            "Exception while featurizing replay {}.\n\n{}"
                            .format(i, fn, traceback.format_exc()))
                    sleep(0.05)
            else:
                logging.warning("Tried to featurize replay {} 1 times,"
                        "giving up".format(fn))
                continue
            data_q.put(res)
            logging.debug("Thread {} loaded replay {}".format(tn, bn))
        logging.debug("Thread {} is done, exiting...".format(tn))

    def stop(self):
        logging.info("Waiting for dataloading threads to exit gracefully...")
        self._done.set()
        while True:
            try:
                self._feeder_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                break

        self._workers = []
        self._feeder_queue = queue.Queue()  # TODO check
        self._data_queue = queue.Queue(self.nthreads)
        self._done = threading.Event()
        self._done.set()
        self._ndone = 0

    def _load(self, *args, read=True, shuffle=True):
        self._uid += 1
        data = []
        if read:
            for fn in args:
                with open(fn, 'r') as f:
                    data += [x.strip() for x in f.readlines()]
            data = [x for x in data if x != '']
        else:
            data = list(chain(*args))
        if shuffle:
            random.shuffle(data)
        for f in data:
            self._feeder_queue.put(path.join(self.args.input, f))
        self._size = len(data)
        logging.debug("Loading {} replays...".format(self._feeder_queue.qsize()))

        self._done.clear()
        for i in range(self.nthreads):
            worker = threading.Thread(
                target=DataLoader._consumer,
                args=(self._feeder_queue, self._data_queue,
                      self._done, self.featurizer),
                name="{}_{}".format(self._uid, i)
            )
            worker.start()
            self._workers.append(worker)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                if (self._done.is_set()
                    or (self._feeder_queue.empty() and
                        not any(x.is_alive() for x in self._workers)
                        and self._data_queue.empty())):
                    self._done.set()
                    raise StopIteration
                data = self._data_queue.get(timeout=1)
                self._ndone += 1
                if (self._ndone % 10) == 0 and self._ndone + len(self._workers) < self._size and self._data_queue.empty():
                    logging.warn("Data queue is empty... Use more dataloading threads!")
                if (self._ndone % min(self._size // 10, 500)) == 0:
                    logging.info("Done {} / {}".format(self._ndone, self._size))
                logging.debug("Data queue has approx size: {}".format(self._data_queue.qsize()))
                return self.postprocess(data)
            except queue.Empty:
                pass


class ChainedDataLoader(object):
    def __init__(self, *args):
        # TODO? init with training
        self.dl = args
        self.dlfn = None
        self.gen = None

    def train(self):
        self.dlfn = [x.train for x in self.dl]

    def valid(self):
        self.dlfn = [x.valid for x in self.dl]

    def test(self):
        self.dlfn = [x.test for x in self.dl]

    def file(self):
        self.dlfn = [x.file for x in self.dl]

    def list(self, data):
        self.dlfn = [x.list for x in self.dl]

    def stop(self):
        logging.info("Stopping a ChainedDataLoader")
        [x.stop() for x in self.dl]

    def __iter__(self):
        def gen():
            for (dl, resetter) in zip(self.dl, self.dlfn):
                resetter()
                for data in dl:
                    yield data
        return gen()
