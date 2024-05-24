import numpy as np
from os.path import isfile
import torchvision, torch, copy
import torch
from torch.utils import data
from pathlib import Path
from utils import get_targeted_classes


def get_labels(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    labels = np.zeros(len(dataset), dtype=int)
    num, max_val = 0, -100000
    
    print('==> Getting label array..')    
    for images, targets in dataloader:
        maxx_img_val = torch.max(images)
        max_val = max(max_val, maxx_img_val)
        labels[num] = targets.item()
        num+=1

    return labels, max_val 


def load_dataset(dataset, root='../data/'):
    # Step 1: Load Transformations and Normalizations
    if dataset in ['CIFAR10','CIFAR100']:
        train_augment = [torchvision.transforms.RandomCrop(32, padding=4), torchvision.transforms.RandomHorizontalFlip()]
        test_augment = []
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    
    elif dataset in ['PCAM']:
        train_augment = [torchvision.transforms.CenterCrop(32), torchvision.transforms.RandomCrop(32, padding=4), torchvision.transforms.RandomHorizontalFlip()]
        test_augment = [torchvision.transforms.CenterCrop(32)]
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    
    elif dataset in ['LFWPeople', 'CelebA', 'DermNet', 'Pneumonia']:
        train_augment = [torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.RandomHorizontalFlip()]
        test_augment = [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)]
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_transforms = torchvision.transforms.Compose(train_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    test_transforms = torchvision.transforms.Compose(test_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

    # Step 2: Load Train, Test and Evaluation Train Sets
    if dataset in ['CIFAR10','CIFAR100']:
        train_set = getattr(torchvision.datasets, dataset)(root=root, train=True, download=True, transform=train_transforms)
        test_set = getattr(torchvision.datasets, dataset)(root=root, train=False, download=True, transform=test_transforms)
        eval_train_set = getattr(torchvision.datasets, dataset)(root=root, train=True, download=True, transform=test_transforms)

    elif dataset in ['PCAM', 'LFWPeople', 'CelebA']:
        train_set = getattr(torchvision.datasets, dataset)(root=root, split='train', download=True, transform=train_transforms)
        test_set = getattr(torchvision.datasets, dataset)(root=root, split='test', download=True, transform=test_transforms)
        eval_train_set = getattr(torchvision.datasets, dataset)(root=root, split='train', download=True, transform=test_transforms)
    
    elif dataset in ['DermNet','Pneumonia']:
        train_set = torchvision.datasets.ImageFolder(root=root+'/'+dataset+'/train', transform=train_transforms)
        test_set = torchvision.datasets.ImageFolder(root=root+'/'+dataset+'/test', transform=test_transforms)
        eval_train_set = torchvision.datasets.ImageFolder(root=root+'/'+dataset+'/train', transform=test_transforms)

    # If found, cache values of labels and max_val else compute them
    if isfile(root+'/'+dataset+'_labels.npy'):
        train_labels = np.load(root+'/'+dataset+'_labels.npy')
        max_val = np.load(root+'/'+dataset+'_maxval.npy')
    else:
        train_labels, max_val = get_labels(eval_train_set)
        np.save(root+'/'+dataset+'_labels.npy', train_labels)
        np.save(root+'/'+dataset+'_maxval.npy', max_val)

    return train_set, eval_train_set, test_set, train_labels, max_val


def manip_dataset(dataset, train_labels, method, manip_set_size, save_dir='../saved_models'):
    assert(method in ['randomlabelswap', 'interclasslabelswap', 'poisoning'])
    manip_idx_path = save_dir+'/'+dataset+'_'+method+'_'+str(manip_set_size)+'_manip.npy'

    if method == 'randomlabelswap' or method == 'poisoning': # Shuffle labels of a selected subset of samples
        if isfile(manip_idx_path):
            manip_idx = np.load(manip_idx_path)
        else:
            manip_idx = np.random.choice(len(train_labels), manip_set_size, replace=False)
            p = Path(save_dir)
            p.mkdir(exist_ok=True)
            np.save(manip_idx_path, manip_idx)
        
        idxes_in_manipidx = copy.deepcopy(manip_idx)
        idxes_in_manipidx.sort()
        
        manip_dict = {}
        for i in range(len(idxes_in_manipidx)):
            manip_dict[idxes_in_manipidx[i]] = train_labels[manip_idx[i]] if method == 'labelrandom' else 0
        
    elif method == 'interclasslabelswap':
        classes = get_targeted_classes(dataset)

        if isfile(manip_idx_path):
            manip_idx = np.load(manip_idx_path)
        else:
            assert(manip_set_size%2==0)
            idx1 = np.asarray(train_labels==classes[0]).nonzero()[0][:manip_set_size//2]
            idx2 = np.asarray(train_labels==classes[1]).nonzero()[0][:manip_set_size//2]
            manip_idx = np.concatenate([idx1, idx2])
            p = Path(save_dir)
            p.mkdir(exist_ok=True)
            np.save(manip_idx_path, manip_idx)

        manip_dict = {}
        for i in range(len(manip_idx)):
            if i < manip_set_size//2:
                manip_dict[manip_idx[i]] = classes[1]
            else:
                manip_dict[manip_idx[i]] = classes[0]

    full_idx = np.arange(len(train_labels))
    untouched_idx = np.setdiff1d(full_idx, manip_idx)
    manip_idx, untouched_idx = torch.from_numpy(manip_idx), torch.from_numpy(untouched_idx)
    return manip_dict, manip_idx, untouched_idx


def get_deletion_set(deletion_size, manip_dict, train_size, dataset, method, save_dir='../saved_models'):
    delete_idx_path = save_dir+'/'+dataset+'_'+method+'_'+str(len(manip_dict))+'_'+str(deletion_size)+'_deletion.npy'
    if isfile(delete_idx_path):
        delete_idx = np.load(delete_idx_path)
    else:
        delete_idx = np.random.choice(np.array(list(manip_dict.keys())), deletion_size, replace=False)
        p = Path(save_dir)
        p.mkdir(exist_ok=True)
        np.save(delete_idx_path, delete_idx)
    
    full_idx = np.arange(train_size)
    retain_idx = np.setdiff1d(full_idx, delete_idx)
    delete_idx, retain_idx = torch.from_numpy(delete_idx), torch.from_numpy(retain_idx)
    return delete_idx, retain_idx


class DatasetWrapper(data.Dataset):
    def __init__(self, dataset, manip_dict, mode='pretrain', corrupt_val=None, corrupt_size=3, delete_idx=None):
        self.dataset = dataset
        self.manip_dict = manip_dict
        self.mode = mode
        if corrupt_val is not None: corrupt_val = torch.from_numpy(corrupt_val)
        self.corrupt_val = corrupt_val
        self.corrupt_size = corrupt_size
        self.delete_idx = delete_idx
        assert(mode in ['pretrain', 'unlearn', 'manip', 'test', 'test_adversarial'])
    
    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)

        if self.mode == 'pretrain': 
            if int(index) in self.manip_dict: # Do nasty things while selecting samples from the manip set
                label = self.manip_dict[int(index)]
                if self.corrupt_val is not None:
                    image[:,-self.corrupt_size:,-self.corrupt_size:] = self.corrupt_val # Have the bottom right corner of the image as the poison
        if self.delete_idx is None:
            self.delete_idx = torch.tensor(list(self.manip_dict.keys()))
        indel = int(index in self.delete_idx)

        if self.mode in ['test', 'test_adversarial']:
            if self.mode == 'test_adversarial':
                image[:,-self.corrupt_size:,-self.corrupt_size:] = self.corrupt_val
            return image, label
        else:
            return image, label, indel
    
    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    train_set, test_set, mean, std, train_labels = load_dataset(dataset='CIFAR10', root='../data/')
    manip_dict = manip_dataset(dataset='CIFAR10', train_labels=train_labels, method='labelrandom', manip_set_size=10000, save_dir='../saved_models')
    #print(manip_dict)
    train_set = DatasetWrapper(train_set, manip_dict, mode='pretrain')
    badt_set = DatasetWrapper(train_set, manip_dict, mode='badt')

    train_set, test_set, mean, std, train_labels = load_dataset(dataset='CIFAR100', root='../data/')
    manip_dict = manip_dataset(dataset='CIFAR100', train_labels=train_labels, method='labelrandom', manip_set_size=1000, save_dir='../saved_models')
    #print(manip_dict)
    train_set = DatasetWrapper(train_set, manip_dict, mode='pretrain')