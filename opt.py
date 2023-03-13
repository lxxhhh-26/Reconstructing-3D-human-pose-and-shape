#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint

__all__ = ['Options']


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_dir', type=str, default='data/', help='path to dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint_totalCapture/', help='path to save checkpoint')
        self.parser.add_argument('--load', type=str, default='', help='path to load a pretrained checkpoint')

        self.parser.add_argument('--test', dest='test', action='store_true', help='test')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--use_hg', dest='use_hg', action='store_true',
                                 help='whether use 2d pose from hourglass')
        self.parser.add_argument('--lr', type=float, default=5e-5)
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--train_batch', type=int, default=64)
        self.parser.add_argument('--test_batch', type=int, default=64)
        self.parser.add_argument('--job', type=int, default=16, help='# subprocesses to use for data loading')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        if self.opt.load:
            if not os.path.isfile(self.opt.load):
                print("{} is not found".format(self.opt.load))
        self.opt.is_train = False if self.opt.test else True
        self.opt.ckpt = ckpt
        self._print()
        return self.opt
