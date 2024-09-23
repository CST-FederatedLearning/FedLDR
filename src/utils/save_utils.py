import glob
import math
import numpy as np
import os
import shutil
import pickle
import torch


def save_checkpoint(state, args, is_best, filename, result):
    path = args.path
    # print(args)
    result_filename = os.path.join(path, 'scores.tsv')
    model_dir = os.path.join(path, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    prev_checkpoint_list = glob.glob(os.path.join(model_dir, 'checkpoint*'))
    if prev_checkpoint_list:
        os.remove(prev_checkpoint_list[0])

    torch.save(state, model_filename)

    with open(result_filename, 'a') as f:
        print(result[-1], file=f)

    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


def load_checkpoint(args, load_best=True):
    path = args.path
    model_dir = os.path.join(path, 'save_models')
    if load_best:
        model_filename = os.path.join(model_dir, 'model_best.pth.tar')
    else:
        model_filename = glob.glob(os.path.join(model_dir, 'checkpoint*'))[0]

    if os.path.exists(model_filename):
        print("=> loading checkpoint '{}'".format(model_filename))
        state = torch.load(model_filename)
        print("=> loaded checkpoint '{}'".format(model_filename))
    else:
        return None

    return state
