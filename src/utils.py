#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from torch.nn.utils.prune import l1_unstructured, random_unstructured, remove


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    # number of fed nodes
    # m = max(int(args.frac * args.num_users), 1)
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def prune_linear(linear, prune_ratio=0.3, method="l1"):
    """Prune a linear layer.

    Modifies the module in-place. We make an assumption that the bias
    is included.

    Parameters
    ----------
    linear : nn.Linear
        Linear module containing a bias.

    prune_ratio : float
        Number between 0 and 1 representing the percentage of weights
        to prune.

    method : str, {"l1", "random"}
        Pruning method to use.
    """
    if method == "l1":
        prune_func = l1_unstructured
    elif method == "random":
        prune_func = random_unstructured
    else:
        raise ValueError

    prune_func(linear, "weight", prune_ratio)
    prune_func(linear, "bias", prune_ratio)
    #remove(linear, 'weight')

def prune_mlp(mlp, prune_ratio=0.3, method="l1"):
    """Prune each layer of the multilayer perceptron.

    Modifies the module in-place. We make an assumption that each
    linear layer has the bias included.

    Parameters
    ----------
    mlp : MLP
        Multilayer perceptron instance.

    prune_ratio : float or list
        Number between 0 and 1 representing the percentage of weights
        to prune. If `list` then different ratio for each
        layer.

    method : str, {"l1", "random"}
        Pruning method to use.
    """
    if isinstance(prune_ratio, float):
        prune_ratios = [prune_ratio] * len(mlp.module_list)
    elif isinstance(prune_ratio, list):
        if len(prune_ratio) != len(mlp.module_list):
            raise ValueError("Incompatible number of prune ratios provided")

        prune_ratios = prune_ratio
    else:
        raise TypeError

    weight_masks = []
    bias_masks = []
    for prune_ratio, linear in zip(prune_ratios, mlp.module_list):
        prune_linear(linear, prune_ratio=prune_ratio, method=method)
        weight_masks.append(list(linear.named_buffers())[0][1])
        bias_masks.append(list(linear.named_buffers())[1][1])
    return weight_masks, bias_masks

def check_pruned_linear(linear):
    """Check if a Linear module was pruned.

    We require both the bias and the weight to be pruned.

    Parameters
    ----------
    linear : nn.Linear
        Linear module containing a bias.

    Returns
    -------
    bool
        True if the model has been pruned.
    """
    params = {param_name for param_name, _ in linear.named_parameters()}
    expected_params = {"weight_orig", "bias_orig"}

    return params == expected_params


def copy_weights_linear(linear_unpruned, linear_pruned):
    """Copy weights from an unpruned model to a pruned model.

    Modifies `linear_pruned` in place.

    Parameters
    ----------
    linear_unpruned : nn.Linear
        Linear model with a bias that was not pruned.

    linear_pruned : nn.Linear
        Linear model with a bias that was pruned.
    """
    assert check_pruned_linear(linear_pruned)
    assert not check_pruned_linear(linear_unpruned)

    with torch.no_grad():
        linear_pruned.weight_orig.copy_(linear_unpruned.weight)
        linear_pruned.bias_orig.copy_(linear_unpruned.bias)


def copy_weights_mlp(mlp_unpruned, mlp_pruned):
    """Copy weights of an unpruned network to a pruned network.

    Modifies `mlp_pruned` in place.

    Parameters
    ----------
    mlp_unpruned : MLP
        MLP model that was not pruned.

    mlp_pruned : MLP
        MLP model that was pruned.
    """
    zipped = zip(mlp_unpruned.module_list, mlp_pruned.module_list)

    for linear_unpruned, linear_pruned in zipped:
        copy_weights_linear(linear_unpruned, linear_pruned)


def remove_reparam_linear(linear_pruned):
    """Make pruning permanent, replace re-parametrization i.e.  weight_orig and weight_mask with just weight

    Modifies `linear_pruned` in place.

    Parameters
    ----------

    linear_pruned : nn.Linear
        Linear model with a bias that was pruned.
    """
    remove(linear_pruned, "weight")
    remove(linear_pruned, "bias")


def remove_reparam_mlp(mlp_pruned):
    """Make pruning permanent, replace re-parametrization i.e.  weight_orig and weight_mask with just weight

    Modifies `mlp_pruned` in place.

    Parameters
    ----------

    mlp_pruned : MLP
        MLP model that was pruned.
    """

    for linear_pruned in mlp_pruned.module_list:
        remove_reparam_linear(linear_pruned)


def compute_stats(mlp):
    """Compute important statistics related to pruning.

    Parameters
    ----------
    mlp : MLP
        Multilayer perceptron.

    Returns
    -------
    dict
        Statistics.
    """
    stats = {}
    total_params = 0
    total_pruned_params = 0

    for layer_ix, linear in enumerate(mlp.module_list):
        assert check_pruned_linear(linear)

        weight_mask = linear.weight_mask
        bias_mask = linear.bias_mask

        params = weight_mask.numel() + bias_mask.numel()
        pruned_params = (weight_mask == 0).sum() + (bias_mask == 0).sum()

        total_params += params
        total_pruned_params += pruned_params

        stats[f"layer{layer_ix}_total_params"] = params
        stats[f"layer{layer_ix}_pruned_params"] = pruned_params
        stats[f"layer{layer_ix}_actual_prune_ratio"] = pruned_params / params

    stats["total_params"] = total_params
    stats["total_pruned_params"] = total_pruned_params
    stats["actual_prune_ratio"] = total_pruned_params / total_params

    return stats