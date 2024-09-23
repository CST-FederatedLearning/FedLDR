import numpy as np

import copy
import os
import gc
import pickle
import time
import sys
import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.model import *
from src.client import *
from src.clustering import *
from src.utils import *
from src.benchmarks import *

# code ref : https://github.com/MMorafah/FedZoo-Bench

if __name__ == '__main__':
    print('-'*40)

    args = args_parser()
    if args.gpu == -1:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(args.gpu) ## Setting cuda on GPU

    args.path = args.logdir + args.alg +'/' + args.dataset + '/' + args.partition + '/'
    if args.partition != 'iid':
        if args.partition == 'iid_qskew':
            args.path = args.path + str(args.iid_beta) + '/'
        else:
            if args.niid_beta.is_integer():
                args.path = args.path + str(int(args.niid_beta)) + '/'+args.model+'/'\
                            '/' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + '/'
            else:
                args.path = args.path + str(args.niid_beta) + '/'+args.model+'/' \
                            '/' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + '/'

    mkdirs(args.path)

    if args.log_filename is None:
        filename='logs_%s.txt' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    else:
        filename='logs_'+args.log_filename+'.txt'

    sys.stdout = Logger(fname=args.path+filename)

    fname=args.path+filename
    fname=fname[0:-4]
    if args.alg == 'solo':
        alg_name = 'SOLO'
        run_solo(args, fname=fname)
    elif args.alg == 'fedavg':
        alg_name = 'FedAvg'
        run_fedavg(args, fname=fname)
    elif args.alg == 'feddr':
        alg_name = 'feddr'
        run_feddr(args, fname=fname)
    else:
        print('Algorithm Does Not Exist')
        sys.exit()
