#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, prune_mlp, prune_linear, copy_weights_mlp, remove_reparam_mlp, compute_stats
from torch.nn.utils.prune import remove


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #if args.gpu_id:
    #    torch.cuda.set_device(args.gpu_id)
    #device = 'cuda' if args.gpu else 'cpu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()
    local_upd_model_copy = copy.deepcopy(global_model)
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    #Pruning set-up
    per_round_prune_ratio = 1 - (1 - args.prune_ratio) ** (
            1 / args.prune_iter
        )

    per_round_prune_ratios = [per_round_prune_ratio] * len(global_model.module_list)
    per_round_prune_ratios[-1] /= 2

    # per_round_max_iter = int(args.max_iter / args.prune_iter)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = np.arange(m)
        np.random.shuffle(idxs_users)

        weights_masks_list, bias_masks_list = [], []
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            local_upd_model = copy.deepcopy(global_model)            

            for iter in range(args.prune_iter):
                w, loss = local_model.update_weights(
                    local_upd_model, global_round=epoch)

                weight_masks, bias_masks = prune_mlp(local_upd_model, per_round_prune_ratios)

                # print("weight masks:", weight_masks)
                # print("bias masks:", bias_masks)
                copy_weights_mlp(local_upd_model_copy, local_upd_model)                
                    
            w, loss = local_model.update_weights(
                    local_upd_model, global_round=epoch)

            weights_masks_list.append(weight_masks)
            bias_masks_list.append(bias_masks)

            stats = compute_stats(local_upd_model)
            print(f'Here are the pruning stats for client {idx} \n')
            print(stats)
                                    
            #print([(key, w[key]) for key in w.keys() ])
            remove_reparam_mlp(local_upd_model)
            
            #local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            #Track pruned weights
            #local_weights.append(copy.deepcopy(local_upd_model.state_dict()))            
            w = local_upd_model.state_dict()
            #print([key for key in w.keys()])
            local_weights.append(copy.deepcopy(w))
                        
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

 
    # Test inference after completion of training
    test_acc, test_loss, test_time = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print("|---- Global Time: {:.6f}".format(test_time))

    # test accuracy for local models
    tmp_local_model = copy.deepcopy(global_model)
    tmp_local_update = LocalUpdate(args=args, dataset=train_dataset,
                                   idxs=user_groups[idx], logger=logger)
    fed_node_acc = []
    fed_node_loss = []
    fed_node_time = []
    w_masks = np.ones(local_weights[0]['layer_input.weight'].shape[0])
    b_masks = np.ones(local_weights[0]['layer_input.bias'].shape[0])
    for i in range(len(weights_masks_list)):

        if args.prune_strat == 2:
            w_masks = weights_masks_list[i]
            b_masks = bias_masks_list[i]

        local_weights[i]['layer_input.weight'] *= w_masks[0]
        local_weights[i]['layer_hidden.weight'] *= w_masks[1]

        local_weights[i]['layer_input.bias'] *= b_masks[0]
        local_weights[i]['layer_hidden.bias'] *= b_masks[1]
        # print("local_weights: ",local_weights[i])
        # print("masked weights: ", masked_weights)

        tmp_local_model.load_state_dict(local_weights[i])
        fed_test_acc, fed_test_loss, fed_test_time = test_inference(args, tmp_local_model, test_dataset)
        print("|---- Test Accuracy fed node {}: {:.2f}%".format(i+1, 100 * fed_test_acc))
        print("|---- Test Time fed node {}: {:.6f}".format(i + 1, fed_test_time))
        fed_node_acc.append(fed_test_acc)
        fed_node_loss.append(fed_test_loss)
        fed_node_time.append(fed_test_time)

    # average accuracy across all fed nodes
    print("|---- Average Test Accuracy across all node: {:.2f}%".format( \
        100 * sum(fed_node_acc)/len(fed_node_acc)))

    print("|---- Min Test Accuracy across node: {:.2f}% for node # {} ".format( \
        100 * np.min(fed_node_acc), np.argmin(fed_node_acc)))

    print("|---- Max Test Accuracy across node: {:.2f}% for node # {} ".format( \
        100 * np.max(fed_node_acc), np.argmax(fed_node_acc)))

    print("|---- Test Accuracy difference between global model and average test "
          " across nodes: {:.2f}%".format( \
        100 * (test_acc - (sum(fed_node_acc)/len(fed_node_acc)))))

    print("|---- Average Test time across all node: {:.6f}".format( \
        sum(fed_node_time)/len(fed_node_time)))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f) 

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
