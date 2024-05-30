import sys
import os
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

import argparse
import pathlib
import numpy as np
import pandas as pd
from src.utils import get_targeted_classes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsvpath', type=str, default='results.tsv', help='path of tsv file (dont add .tsv)')
    parser.add_argument('--dirpath', type=str, default='./', help='root directory of saved results')
    return parser.parse_args()

def parse_pretrain_dirname(dirname, preargnamelist):
    print(dirname)
    arglist = dirname.split('_')
    
    assert(len(arglist) == len(preargnamelist))
    pretrain_args = {}
    for idx, name in enumerate(preargnamelist):
        pretrain_args[name] = arglist[idx]
    return pretrain_args

def parse_unlearn_dirname(dirname, unargnamelist):
    print("dirname:", dirname)
    arglist = dirname.split('_')
    assert(len(arglist) >= 2)
    
    un_args = dict((arg, '') for arg in unargnamelist)
    print(un_args)
    if arglist[0] == 'Naive':
        un_args['unlearn_method'], un_args['exp_name'] = arglist[0], arglist[1]
        un_args['deletion_size'] = 0
        return un_args
    un_args['deletion_size'], un_args['unlearn_method'], un_args['exp_name'] = arglist[0], arglist[1], arglist[2]
    if arglist[1] in ['EU', 'CF', 'Mixed', 'Scrub', 'BadT', 'SSD', 'AscentLearn', 'ScrubNew', 'ALnew']:
        assert(len(arglist) >= 5)
        un_args['train_iters'], un_args['k'] = arglist[3], arglist[4] 
    if arglist[1] in ['Mixed']:
        assert(len(arglist) >= 6)
        un_args['factor'] = arglist[5]
    if arglist[1] in ['AscentLearn', 'ALnew']:
        assert(len(arglist) >= 6)
        un_args['ascLRscale'] = arglist[5]
    if arglist[1] == 'InfRe':
        assert(len(arglist) >= 8)
        un_args['msteps'], un_args['rsteps'], un_args['ascLRscale'] = arglist[5], arglist[6], arglist[7]
    if arglist[1] == 'Scrub':
        assert(len(arglist) >= 8)
        un_args['kd_T'], un_args['alpha'], un_args['msteps'] = arglist[5:8]
    if arglist[1] == 'ScrubNew':
        assert(len(arglist) >= 8)
        un_args['kd_T'], un_args['alpha'], un_args['ascLRscale'] = arglist[5:8]
    if arglist[1] == 'SSD':
        assert(len(arglist) >= 7)
        un_args['SSDdampening'], un_args['selectwt'] = arglist[5], arglist[6]
    return un_args

def compute_accuracy(preds, y):
    return np.equal(np.argmax(preds, axis=1), y).mean()

def parse_unpath(un_path, pre_args, un_args, args, headers):
    ret = dict((key, '') for key in headers)
    ret.update(pre_args)
    ret.update(un_args)

    tr_preds = np.load(un_path + f'/preds_train.npy')
    tr_y = np.load(un_path + f'/targetstrain.npy')
    te_preds = np.load(un_path + f'/preds_test.npy')
    te_y = np.load(un_path + f'/targetstest.npy')
    un_time = np.load(un_path + f'/unlearn_time.npy')
    ret['unlearn_time'] = un_time
    forget_idx = np.load(args.dirpath+'/'+pre_args['dataset']+'_'+pre_args['dataset_method']+'_'+pre_args['forget_set_size']+'_manip.npy')
    if un_args['deletion_size'] != 0:
        delete_idx = np.load(args.dirpath+'/'+pre_args['dataset']+'_'+pre_args['dataset_method']+'_'+pre_args['forget_set_size']+'_'+un_args['deletion_size']+'_deletion.npy')
    ret['train_clean_acc'] = compute_accuracy(tr_preds, tr_y) 
    delete_acc, delete_err = 0.0, 101.0
    if pre_args['dataset_method'] == 'poisoning':
        tr_adv_preds = np.load(un_path + f'/preds_adv_train.npy')
        tr_adv_y = np.load(un_path + f'/targetsadv_train.npy')
        tr_wrong = np.zeros(tr_adv_y.shape)
        te_adv_preds = np.load(un_path + f'/preds_adv_test.npy')
        te_adv_y = np.load(un_path + f'/targetsadv_test.npy')
        forget_acc = compute_accuracy(tr_adv_preds[forget_idx], tr_adv_y[forget_idx])
        if un_args['deletion_size'] != 0: 
            delete_err = compute_accuracy(tr_adv_preds[delete_idx], tr_wrong[delete_idx])
            delete_acc = compute_accuracy(tr_adv_preds[delete_idx], tr_adv_y[delete_idx])
        test_acc = compute_accuracy(te_adv_preds, te_adv_y)
        forget_clean_acc = compute_accuracy(tr_preds[forget_idx], tr_y[forget_idx])
        test_clean_acc = compute_accuracy(te_preds, te_y)
        print("forget_acc:", forget_acc)
        print("test_acc:", test_acc)
        print("forget_clean_acc:", forget_clean_acc)
        print("test_clean_acc:", test_clean_acc)
        ret['delete_acc'], ret['delete_err'], ret['manip_acc'], ret['test_acc'], ret['manip_clean_acc'], ret['test_clean_acc'] =\
         delete_acc, delete_err, forget_acc, test_acc, forget_clean_acc, test_clean_acc


    if pre_args['dataset_method'] == 'dontuse':
        forget_acc = compute_accuracy(tr_preds[forget_idx], tr_y[forget_idx])
        test_acc = compute_accuracy(te_preds, te_y)
        print("forget_acc:", forget_acc)
        print("test_acc:", test_acc)
        ret['forget_acc'], ret['test_acc'] = forget_acc, test_acc

    if pre_args['dataset_method'] == 'interclasslabelswap':
        classes = get_targeted_classes(pre_args['dataset'])
        te_class_idxes = np.concatenate((np.nonzero(te_y == classes[0]), np.nonzero(te_y == classes[1])), axis=1).squeeze()
        retain_idxes =np.setdiff1d(np.arange(len(te_y)), te_class_idxes)
        forget_acc = compute_accuracy(tr_preds[forget_idx], tr_y[forget_idx])
        tr_wrong = tr_y
        tr_wrong[tr_y == classes[0]] = classes[1]
        tr_wrong[tr_y == classes[1]] = classes[0]
        if un_args['deletion_size'] != 0: 
            delete_acc = compute_accuracy(tr_preds[delete_idx], tr_y[delete_idx])
            delete_err = compute_accuracy(tr_preds[delete_idx], tr_wrong[delete_idx])
        test_acc = compute_accuracy(te_preds[te_class_idxes], te_y[te_class_idxes])
        test_retain_acc = compute_accuracy(te_preds[retain_idxes], te_y[retain_idxes])
        print(forget_acc, test_acc, test_retain_acc) 
        ret['delete_acc'], ret['delete_err'], ret['manip_acc'], ret['test_acc'], ret['test_retain_acc']  = delete_acc, delete_err, forget_acc, test_acc, test_retain_acc

    return ret


if __name__ == '__main__':
    # datasets = ['CIFAR10', 'CIFAR100', 'PCAM', 'Pneumonia', 'DermaNet']
    # model = ['resnet9', 'resnetwide28x10']
    # dataset_method = ['labelrandom', 'labeltargeted', 'poisoning']
    # unlearn_method = ['Naive', 'EU', 'CF', 'Mixed', 'Scrub']
    preargnamelist = ['dataset', 'model', 'dataset_method', 'forget_set_size', 'patch_size', 'pretrain_iters', 'pretrain_lr']
    unargnamelist = ['unlearn_method', 'exp_name', 'train_iters', 'k', 'factor', 'kd_T', 'gamma', 'alpha', 'msteps']
    metricslist = ['delete_acc', 'delete_err', 'manip_acc', 'test_acc', 'manip_clean_acc', 'test_clean_acc', 'test_retain_acc']
    headers = preargnamelist + unargnamelist + metricslist

    args = parse_args()
    rows = []
    for dirname in next(os.walk(args.dirpath))[1]:
        print(dirname)
        pre_args = parse_pretrain_dirname(dirname, preargnamelist)
        pretrain_path = os.path.join(args.dirpath, dirname)
        for undirname in next(os.walk(pretrain_path))[1]:
            print(dirname, undirname)
            un_args = parse_unlearn_dirname(undirname, unargnamelist)
            un_path = os.path.join(pretrain_path, undirname)
            row = parse_unpath(un_path, pre_args, un_args, args, headers)
            rows.append(row)
    
    df = pd.DataFrame.from_records(rows)
    df.replace('', 'Null', inplace=True)
    df.to_csv(args.tsvpath, sep='\t', index=False)


    
