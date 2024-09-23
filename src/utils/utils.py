import sys
import os

# from src.models.lenet5 import LeNet_DR
from torchvision import models

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from src.data import *
from src.models import *
from src.utils import *
from src.models.resnet import *
# from src.models.lenet5 import *

import numpy as np

import copy
import gc

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.init as init
from thop import profile


class Logger(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_classes(dataset):
    if dataset in ['cifar100']:
        return 100
    elif dataset in ['fmnist', 'cifar10', 'cinic10']:
        return 10
    elif dataset == 'stl10':
        return 100
    elif dataset == 'tinyimagenet':
        return 200
    elif dataset == 'ham10000':
        return 7
    elif dataset == 'medmnist':
        return 6
    elif dataset == 'covid19':
        return 4
    else:
        print("utils_get_classes: dataset not supported yet")
        sys.exit()


def AvgWeights(w, weight_avg=None):
    """
    Federated averaging
    :param w: list of client model parameters
    :return: updated server model parameters
    """
    if weight_avg == None:
        weight_avg = [1 / len(w) for i in range(len(w))]

    w_avg = copy.deepcopy(w[0])
    device = list(w[0].values())[0].get_device()
    # device =
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].to(device) * weight_avg[0]

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].to(device) + w[i][k].to(device) * weight_avg[i]
        # w_avg[k] = torch.div(w_avg[k].to(device), len(w))

    return w_avg


def AvgWeights_dis(w, w_pre=None, weight_avg=None):
    """
    Federated averaging
    :param w: list of client model parameters
    :return: updated server model parameters
    """
    if w_pre is not None:
        wr = copy.deepcopy(w[0])
        cos = torch.nn.CosineSimilarity(dim=-1)

        for k in wr.keys():
            for i in range(1, len(w)):
                wr[k] = (cos(torch.Tensor.float(w[i][k]), torch.Tensor.float(w_pre[i][k])))

        return wr, w
    else:
        if weight_avg == None:
            weight_avg = [1 / len(w) for i in range(len(w))]

        w_avg = copy.deepcopy(w[0])
        device = list(w[0].values())[0].get_device()
        # device =
        for k in w_avg.keys():
            w_avg[k] = w_avg[k].to(device) * weight_avg[0]

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] = w_avg[k].to(device) + w[i][k].to(device) * weight_avg[i]
            # w_avg[k] = torch.div(w_avg[k].to(device), len(w))
        return w_avg, w

    # return w_avg, w


class SimpleCNN_3:
    pass


def get_distr_model(args, same_init=True, class_num=10):
    users_model = []

    for i in range(-1, args.num_users):
        if args.model == "mlp":
            continue
        elif args.model == "lenet5":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = LeNet5_MNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset == 'celeba':
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2).to(args.device)
        elif args.model == "simple-cnn-3":
            if args.dataset == 'cifar100':
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120 * 3, 84 * 3], output_dim=100).to(
                    args.device)
            if args.dataset == 'tinyimagenet':
                net = LeNet5_TinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120 * 3, 84 * 3],
                                            output_dim=200).to(args.device)
        elif args.model == "vgg9":
            if args.dataset in ("mnist", 'femnist'):
                # net = ModerateCNNMNIST().to(args.device)
                pass
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                # net = ModerateCNN().to(args.device)
                pass
            elif args.dataset == 'celeba':
                # net = ModerateCNN(output_dim=2).to(args.device)
                pass
        elif args.model == 'resnet9':
            if args.dataset in ['cifar100']:
                net = ResNet9(in_channels=3, num_classes=class_num)
            elif args.dataset == 'cifar10' or args.dataset == 'cinic10':
                net = ResNet9(in_channels=3, num_classes=class_num)
            elif args.dataset == 'fmnist':
                net = ResNet9(in_channels=1, num_classes=class_num)
            elif args.dataset == 'stl10':
                net = ResNet9(in_channels=3, num_classes=class_num, dim=4608)
            elif args.dataset == 'tinyimagenet':
                net = ResNet9(in_channels=3, num_classes=class_num, dim=512 * 2 * 2)
            elif args.model == "vgg16":
                net = vgg16().to(args.device)
        elif args.model == 'resnet18':
            net = ResNet18(num_classes=class_num).to(args.device)
        elif args.model == 'resnet50':
            net = ResNet50(num_classes=class_num).to(args.device)
        elif args.model == "resnet101":
            net = ResNet101(num_classes=class_num).to(args.device)
        else:
            print("2. alg not supported yet")
            sys.exit()

        if i == -1:
            net_glob = copy.deepcopy(net)
            net_glob.apply(weight_init)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            if same_init:
                users_model[i].load_state_dict(initial_state_dict)

    return users_model


def get_client_id_model(args, same_init=True):
    users_model = []
    class_num = get_classes(args.dataset)
    dis_model = args.dis_model
    net = None

    for i in range(-1, args.num_users):
        if dis_model == "mlp":
            if args.dataset == 'fmnist':
                net = MLP_DR_1(output_dim=args.num_users)
            else:
                net = MLP_DR(output_dim=args.num_users)
        elif dis_model == "lenet5":
            if args.dataset in (  "svhn"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(args.device)
            if args.dataset in ("cifar10","cinic10","cifar100"):
                net = LeNet_DR(output_dim=args.num_users).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = LeNet5_MNIST(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(args.device)
            elif args.dataset == 'celeba':
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(args.device)
        elif dis_model =="simple-cnn-3":
            if args.dataset == 'cifar100':
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=args.num_users).to(args.device)
            if args.dataset == 'tinyimagenet':
                net = LeNet5_TinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120 * 3, 84 * 3],
                                            output_dim=args.num_users).to(args.device)
        elif dis_model == "vgg11":
            if args.dataset in ("mnist", 'femnist'):
                # net = ModerateCNNMNIST().to(args.device)
                net = VGG_DR('VGG11', output_dim=args.num_users).to(args.device)
                pass
            elif args.dataset in ("cifar10", "cinic10", "svhn", "cifar100"):
                # print("in moderate cnn")
                # net = ModerateCNN().to(args.device)
                net = VGG_DR('VGG11', input_size=32, output_dim=args.num_users).to(args.device)
                pass
            elif args.dataset == 'celeba':
                # net = ModerateCNN(output_dim=2).to(args.device)
                net = VGG_DR('VGG11', output_dim=args.num_users).to(args.device)
                pass
        elif dis_model == 'resnet9':
            if args.dataset in ['cifar100']:
                net = ResNet9(in_channels=3, num_classes=args.num_users)
            elif args.dataset == 'cifar10' or args.dataset == 'cinic10':
                net = ResNet9(in_channels=3, num_classes=args.num_users)
            elif args.dataset == 'fmnist':
                net = ResNet9(in_channels=1, num_classes=args.num_users)
            elif args.dataset == 'stl10':
                net = ResNet9(in_channels=3, num_classes=args.num_users, dim=4608)
            elif args.dataset == 'tinyimagenet':
                net = ResNet9(in_channels=3, num_classes=args.num_users, dim=512 * 2 * 2)
            elif args.dataset in ['medmnist']:
                net = ResNet9(in_channels=3, num_classes=6)
            elif args.dataset in ['covid19']:
                net = ResNet9(in_channels=3, num_classes=4,dim=25088)
            elif dis_model == "resnet":
                net = ResNet50_cifar10().to(args.device)
            elif dis_model == "vgg16":
                net = vgg16().to(args.device)
        elif dis_model == 'resnet18':
            net = ResNet18(num_classes=class_num).to(args.device)
        elif dis_model == 'resnet50':
            net = ResNet50(num_classes=class_num).to(args.device)
        elif dis_model == "resnet101":
            net = ResNet101(num_classes=class_num).to(args.device)
        else:
            print("3. alg not supported yet")
            sys.exit()

        if i == -1:
            net_glob = copy.deepcopy(net)
            net_glob.apply(weight_init)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            if same_init:
                users_model[i].load_state_dict(initial_state_dict)

    return net


def get_user_clients_model(args, same_init=True):
    users_model = []
    class_num = get_classes(args.dataset)
    dis_model = args.dis_model
    net = None

    for i in range(-1, args.num_users):
        if dis_model == "mlp":
            if args.dataset == 'fmnist':
                net = MLP_DR_1(output_dim=args.num_users)
            else:
                net = MLP_DR(output_dim=args.num_users)
        elif dis_model == "lenet5":
            if args.dataset in (  "svhn"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(args.device)
            if args.dataset in ("cifar10","cinic10","cifar100"):
                net = LeNet_DR(output_dim=args.num_users).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = LeNet5_MNIST(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(args.device)
            elif args.dataset == 'celeba':
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(args.device)
        elif dis_model =="simple-cnn-3":
            if args.dataset == 'cifar100':
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=args.num_users).to(args.device)
            if args.dataset == 'tinyimagenet':
                net = LeNet5_TinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120 * 3, 84 * 3],
                                            output_dim=args.num_users).to(args.device)
        elif dis_model == "vgg11":
            if args.dataset in ("mnist", 'femnist'):
                # net = ModerateCNNMNIST().to(args.device)
                net = VGG_DR('VGG11', output_dim=args.num_users).to(args.device)
                pass
            elif args.dataset in ("cifar10", "cinic10", "svhn", "cifar100"):
                # print("in moderate cnn")
                # net = ModerateCNN().to(args.device)
                net = VGG_DR('VGG11', input_size=32, output_dim=args.num_users).to(args.device)
                pass
            elif args.dataset == 'celeba':
                # net = ModerateCNN(output_dim=2).to(args.device)
                net = VGG_DR('VGG11', output_dim=args.num_users).to(args.device)
                pass
        elif dis_model == 'resnet9':
            if args.dataset in ['cifar100']:
                net = ResNet9(in_channels=3, num_classes=args.num_users)
            elif args.dataset == 'cifar10' or args.dataset == 'cinic10':
                net = ResNet9(in_channels=3, num_classes=args.num_users)
            elif args.dataset == 'fmnist':
                net = ResNet9(in_channels=1, num_classes=args.num_users)
            elif args.dataset == 'stl10':
                net = ResNet9(in_channels=3, num_classes=args.num_users, dim=4608)
            elif args.dataset == 'tinyimagenet':
                net = ResNet9(in_channels=3, num_classes=args.num_users, dim=512 * 2 * 2)
            elif args.dataset in ['medmnist']:
                net = ResNet9(in_channels=3, num_classes=6)
            elif args.dataset in ['covid19']:
                net = ResNet9(in_channels=3, num_classes=4,dim=25088)
            elif dis_model == "resnet":
                net = ResNet50_cifar10().to(args.device)
            elif dis_model == "vgg16":
                net = vgg16().to(args.device)
        elif dis_model == 'resnet18':
            net = ResNet18(num_classes=class_num).to(args.device)
        elif dis_model == 'resnet50':
            net = ResNet50(num_classes=class_num).to(args.device)
        elif dis_model == "resnet101":
            net = ResNet101(num_classes=class_num).to(args.device)
        else:
            print("3. alg not supported yet")
            sys.exit()

        if i == -1:
            net_glob = copy.deepcopy(net)
            net_glob.apply(weight_init)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            if same_init:
                users_model[i].load_state_dict(initial_state_dict)

    return users_model

def get_imagesize(args):
    if args.dataset == 'cifar10' or args.dataset == 'cinic10':
        args.num_classes = 10
        args.image_size = (32, 32)
    elif args.dataset == 'fmnist':
        args.num_classes = 10
        args.image_size = (28, 28)
    elif args.dataset == 'ham10000':
        args.num_classes = 7
        args.image_size = (32, 32)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.image_size = (32, 32)
    elif args.dataset == 'medmnist':
        args.num_classes = 6
        args.image_size = (28, 28)
    elif args.dataset == 'covid19':
        args.num_classes = 4
        args.image_size = (224, 224)
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        if 'effnetb4' in args.arch:
            args.image_size = (224, 224)
        else:
            args.image_size = (64, 64)
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        args.image_size = (64, 64)
    elif args.dataset == 'stl10':
        args.num_classes = 10
        args.image_size = (96, 96)
    elif args.dataset == 'sst2':
        args.num_classes = 2
        args.image_size = (1, 64)
    elif args.dataset == 'ag_news':
        args.num_classes = 4
        args.image_size = (1, 64)
    else:
        print("4. data not supported yet")
        raise NotImplementedError


def get_inchannels(dataset):
    if dataset in ("mnist", 'femnist', 'fmnist'):
        return 1
    return 3


def get_models(args, dropout_p=0.5, same_init=True):
    users_model = []
    class_num = get_classes(args.dataset)
    get_imagesize(args)
    H, W = args.image_size[0], args.image_size[1]

    for i in range(-1, args.num_users):
        if args.model == "mlp":
            if args.dataset == 'fmnist':
                net = MLP_DR_1(output_dim=args.num_users)
            else:
                net = MLP_DR(output_dim=args.num_users)
        elif args.model == "lenet5":
            if args.dataset in ("svhn"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
                # net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(args.device)
            if args.dataset in ("cifar10", "cinic10", "cifar100"):
                net = LeNet_DR(output_dim=10).to(args.device)
                # net = LeNet_DR(output_dim=args.num_users).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = LeNet5_MNIST(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(
                    args.device)
                # net = LeNet5_MNIST(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(
                #     args.device)
            elif args.dataset == 'celeba':
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
                # net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_users).to(args.device)
        elif args.model == "simple-cnn-3":
            if args.dataset == 'cifar100':
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120 * 3, 84 * 3],
                                  output_dim=args.num_users).to(args.device)
            if args.dataset == 'tinyimagenet':
                net = LeNet5_TinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120 * 3, 84 * 3],
                                            output_dim=args.num_users).to(args.device)
        elif args.model == "vgg11":
            if args.dataset in ("mnist", 'femnist'):
                # net = ModerateCNNMNIST().to(args.device)
                net = VGG_DR('VGG11', output_dim=args.num_users).to(args.device)
                pass
            elif args.dataset in ("cifar10", "cinic10", "svhn", "cifar100"):
                # print("in moderate cnn")
                # net = ModerateCNN().to(args.device)
                net = VGG_DR('VGG11', output_dim=args.num_users).to(args.device)
                pass
            elif args.dataset == 'celeba':
                # net = ModerateCNN(output_dim=2).to(args.device)
                net = VGG_DR('VGG11', output_dim=args.num_users).to(args.device)
                pass
        elif args.model == 'resnet9':
            if args.dataset in ['cifar100']:
                net = ResNet9(in_channels=3, num_classes=100)
            elif args.dataset == 'cifar10' or args.dataset == 'cinic10':
                net = ResNet9(in_channels=3, num_classes=10)
            elif args.dataset in ['fmnist']:
                net = ResNet9(in_channels=1, num_classes=10)
            elif args.dataset == 'stl10':
                net = ResNet9(in_channels=3, num_classes=100, dim=4608)
            elif args.dataset == 'tinyimagenet':
                net = ResNet9(in_channels=3, num_classes=200, dim=512 * 2 * 2)
            elif args.dataset in ['ham10000']:
                net = ResNet9(in_channels=3, num_classes=7)
            elif args.dataset in ['medmnist']:
                net = ResNet9(in_channels=3, num_classes=6)
            # elif args.dataset in ['covid19']:
            #     # net = models.resnet18()
            #     net = ResNet9(in_channels=3, num_classes=4)
        elif args.model == "resnet":
            net = ResNet50_cifar10().to(args.device)
        elif args.model == "vgg16":
            net = vgg16().to(args.device)
        elif args.model == 'resnet18':
            net = ResNet18(num_classes=class_num).to(args.device)
        elif args.model == 'resnet34':
            net = ResNet34(num_classes=class_num).to(args.device)
        elif args.model == 'resnet50':
            net = ResNet50(num_classes=class_num).to(args.device)
        elif args.model == "resnet101":
            net = ResNet101(num_classes=class_num).to(args.device)
        else:
            print("1. alg not supported yet")
            sys.exit()

        if i == -1:
            net_glob = copy.deepcopy(net)
            net_glob.apply(weight_init)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            if same_init:
                users_model[i].load_state_dict(initial_state_dict)

    # in_channels = get_inchannels(args.dataset)
    # data = Variable(torch.zeros(1, in_channels, H, W).to(args.device))
    # model_eval = copy.deepcopy(net)
    # model_eval.eval()
    # flops, params = profile(model_eval, (data,))
    # print(f'flops:{flops}, params:{params}')

    return users_model, net_glob, initial_state_dict


class CIFAR10_SuperClass_NIID_DIR:
    pass


class CIFAR10_SuperClass_NIID:
    pass


class CIFAR10_SuperClass_Old_NIID_DIR:
    pass


class CIFAR10_SuperClass_Old_NIID:
    pass


def get_clients_data(args):
    if args.partition[0:2] == 'sc':
        if args.dataset == 'cifar10':
            if args.partition == 'sc_niid_dir':
                print('Loading CIFAR10 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = CIFAR10_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)

            elif args.partition[0:7] == 'sc_niid':
                print('Loading CIFAR10 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = CIFAR10_SuperClass_NIID(train_ds_global, test_ds_global, num, args)

            elif args.partition == 'sc_old_niid_dir':
                print('Loading CIFAR10 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = CIFAR10_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)

            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading CIFAR10 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = CIFAR10_SuperClass_Old_NIID(train_ds_global, test_ds_global, num, args)

        elif args.dataset == 'cifar100':
            if args.partition == 'sc_niid_dir':
                print('Loading CIFAR100 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = CIFAR100_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)

            elif args.partition[0:7] == 'sc_niid':
                print('Loading CIFAR100 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = CIFAR100_SuperClass_NIID(train_ds_global, test_ds_global, args)

            elif args.partition == 'sc_old_niid_dir':
                print('Loading CIFAR100 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = CIFAR100_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)

            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading CIFAR100 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = CIFAR100_SuperClass_Old_NIID(train_ds_global, test_ds_global, args)

        elif args.dataset == 'stl10':
            if args.partition == 'sc_niid_dir':
                print('Loading STL10 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = STL10_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)

            elif args.partition[0:7] == 'sc_niid':
                print('Loading STL10 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = STL10_SuperClass_NIID(train_ds_global, test_ds_global, num, args)

            elif args.partition == 'sc_old_niid_dir':
                print('Loading STL10 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = STL10_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)

            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading STL10 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = STL10_SuperClass_Old_NIID(train_ds_global, test_ds_global, num, args)

        elif args.dataset == 'fmnist':
            if args.partition == 'sc_niid_dir':
                pass
            elif args.partition[0:7] == 'sc_niid':
                print('Loading FMNIST SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                    = FMNIST_SuperClass_NIID(train_ds_global, test_ds_global, args)
    else:
        print(f'Loading {args.dataset}, {args.partition} for all clients')

        partitions_train, partitions_test, \
            partitions_train_stat, partitions_test_stat = partition_data(args.dataset,
                                                                         args.datadir, args.partition, args.num_users,
                                                                         niid_beta=args.niid_beta,
                                                                         iid_beta=args.iid_beta)

    return partitions_train, partitions_test, partitions_train_stat, partitions_test_stat


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    return


def eval_test(net, args, ldr_test):
    net.to(args.device)
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in ldr_test:
            data, target = data.to(args.device), target.to(args.device)
            target = target.type(torch.LongTensor).to(args.device)

            output = net(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(ldr_test.dataset)
    accuracy = 100. * correct / len(ldr_test.dataset)
    return test_loss, accuracy
