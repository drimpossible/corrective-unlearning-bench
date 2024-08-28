import torch, torchmetrics, tqdm, copy, time
from utils import LinearLR, unlearn_func, ssd_tuning, distill_kl_loss, compute_accuracy
from torch.cuda.amp import autocast
import numpy as np
import resnet
from torch.cuda.amp import GradScaler
from pathlib import Path
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from sklearn.decomposition import PCA
import sys
sys.path.append('/data/jiawei_li/corrective-unlearning-bench/src')
# import pytorch_influence_functions as ptif
from torch.utils.data import Dataset, DataLoader
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from typing import Tuple
from torchvision import transforms
from kronfluence.arguments import FactorArguments
from utils import get_targeted_classes
from dataset import DatasetWrapper

class DataSetWrapper(Dataset):
    """ Wrap the dataset to return only images and targets, as expected by ptif """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data[0], data[1]  

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return idx, data, target

class CollectedDataset(Dataset):
    """Aggregates all data into a single dataset from a DataLoader."""
    def __init__(self, loader):
        self.data = []
        self.targets = []
        self.additional_info = []

        first_batch = next(iter(loader))
        has_additional_info = len(first_batch) == 3

        for batch in loader:
            self.data.append(batch[0])
            batch_targets = batch[1]
            if batch_targets.ndim == 0:
                batch_targets = batch_targets.unsqueeze(0) 
            self.targets.append(batch_targets)
            if has_additional_info:
                self.additional_info.append(batch[2])
        
        # Concatenate lists into tensors
        self.data = torch.cat(self.data, dim=0)
        self.targets = torch.cat(self.targets, dim=0)
        if has_additional_info:
            self.additional_info = torch.cat(self.additional_info, dim=0)
        else:
            self.additional_info = None  # Handle cases without additional info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.additional_info is not None:
            return self.data[index], self.targets[index], self.additional_info[index]
        else:
            return self.data[index], self.targets[index]

class CustomDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

class Naive():
    def __init__(self, opt, model, prenet=None):
        self.opt = opt
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.set_model(model, prenet)
        self.save_files = {'train_top1':[], 'val_top1':[], 'train_time_taken':0}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.max_lr, momentum=0.9, weight_decay=self.opt.wd)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.train_iters*1.25, warmup_epochs=self.opt.train_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        if self.opt.device == 'cuda':
            self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.opt.num_classes).cuda()
        else:
            self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.opt.num_classes)
        self.scaler = GradScaler()


    def set_model(self, model, prenet=None):
        self.prenet = None
        self.model = model
        if self.opt.device == 'cuda':
            self.model.cuda()


    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        loss = F.cross_entropy(output, target)
        self.top1(output, target)
        return loss


    def train_one_epoch(self, loader):
        self.model.train()
        self.top1.reset()

        for (images, target, infgt) in tqdm.tqdm(loader):
            images, target, infgt = images.cuda(), target.cuda(), infgt.cuda()
            with autocast():
                self.optimizer.zero_grad()
                loss = self.forward_pass(images, target, infgt)
                print("Loss:", loss.item())
            self.scaler.scale(loss).backward()
            found_nan_or_inf = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        found_nan_or_inf = True
                        print("NaN or Inf found in gradients")
                        break

            if not found_nan_or_inf:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                print("Skipping optimizer step due to NaN or Inf in gradients")
            self.scheduler.step()
            self.curr_step += 1
            if self.curr_step > self.opt.train_iters:
                break

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files['train_top1'].append(top1)
        print(f'Step: {self.curr_step} Train Top1: {top1:.3f}')
        return


    def eval(self, loader, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        if save_preds:
            preds, targets = [], []

        with torch.no_grad():
            for (images, target) in tqdm.tqdm(loader):
                with autocast():
                    images, target = images.cuda(), target.cuda()
                    output = self.model(images) if self.prenet is None else self.model(self.prenet(images))
                self.top1(output, target)
                if save_preds:
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        top1 = self.top1.compute().item()
        self.top1.reset()
        if not save_preds: print(f'Step: {self.curr_step} Val Top1: {top1*100:.2f}')
        
        if save_model:
            self.save_files['val_top1'].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return


    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        while self.curr_step < self.opt.train_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
        return


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+self.opt.unlearn_method+'_'+self.opt.exp_name
        if self.opt.unlearn_method != 'Naive': 
            self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        return


    def compute_and_save_results(self, train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader):
        self.get_save_prefix()
        print(self.unlearn_file_prefix)
        if not exists(self.unlearn_file_prefix):
            makedirs(self.unlearn_file_prefix)

        torch.save(self.best_model.state_dict(), self.unlearn_file_prefix+'/model.pth')
        np.save(self.unlearn_file_prefix+'/train_top1.npy', self.save_files['train_top1'])
        np.save(self.unlearn_file_prefix+'/val_top1.npy', self.save_files['val_top1'])
        np.save(self.unlearn_file_prefix+'/unlearn_time.npy', self.save_files['train_time_taken'])
        self.model = self.best_model.cuda()

        print('==> Completed! Unlearning Time: [{0:.3f}]\t'.format(self.save_files['train_time_taken']))
        
        for loader, name in [(train_test_loader, 'train'), (test_loader, 'test'), (adversarial_train_loader, 'adv_train'), (adversarial_test_loader, 'adv_test')]:
            if loader is not None:
                preds, targets = self.eval(loader=loader, save_preds=True)
                np.save(self.unlearn_file_prefix+'/preds_'+name+'.npy', preds)
                np.save(self.unlearn_file_prefix+'/targets'+name+'.npy', targets)
        return


class ApplyK(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
    

    def set_model(self, model, prenet):
        prenet, model = self.divide_model(model, k=self.opt.k, model_name=self.opt.model)
        model = unlearn_func(model=model, method=self.opt.unlearn_method, factor=self.opt.factor, device=self.opt.device) 
        self.model = model
        self.prenet = prenet
        self.model.cuda()
        if self.prenet is not None: self.prenet.cuda().eval()


    def divide_model(self, model, k, model_name):
        if k == -1: # -1 means retrain all layers
            net = model
            prenet = None
            return prenet, net

        if model_name == 'resnet9':
            assert(k in [1,2,4,5,7,8])
            mapping = {1:6, 2:5, 4:4, 5:3, 7:2, 8:1}
            dividing_part = mapping[k]
            all_mods = [model.conv1, model.conv2, model.res1, model.conv3, model.res2, model.conv4, model.fc] 
            prenet = torch.nn.Sequential(*all_mods[:dividing_part])
            net = torch.nn.Sequential(*all_mods[dividing_part:])

        elif model_name == 'resnetwide28x10':
            assert(k in [1,3,5,7,9,11,13,15,17,19,21,23,25])
            all_mods = [model.conv1, model.layer1, model.layer2, model.layer3, model.norm, model.fc]
            mapping = {1:5, 9:3, 17:2, 25:1}
    
            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                vals = list(mapping.keys())
                for val in vals:
                    if val > k:
                        sel_idx = val
                        break
                layer = mapping[sel_idx]
                prenet_list = all_mods[:layer]
                prenet_additions = list(all_mods[layer][:int(4-(((k-1)//2)%4))])
                prenet = torch.nn.Sequential(*(prenet_list+prenet_additions))
                net_list = list(all_mods[layer][int(4-(((k-1)//2)%4)):])
                net_additions = all_mods[layer+1:]
                net = torch.nn.Sequential(*(net_list+net_additions))

        elif model_name == 'vitb16':
            assert(k in [1,2,3,4,5,6,7,8,9,10,11,12,13])
            all_mods = [model.patch_embed, model.blocks, model.norm, model.head]
            mapping = {1:3, 13:1}

            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                prenet = [model.patch_embed]
                k = 13-k
                prenet += [model.blocks[:k]]
                prenet = torch.nn.Sequential(*prenet)
                net = [model.blocks[k:], model.norm, model.head]
                net = torch.nn.Sequential(*net)
                
        prenet.to(self.opt.device)
        net.to(self.opt.device)
        return prenet, net


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)

        return 


class Scrub(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        self.og_model.cuda().eval()
    

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)

        with torch.no_grad():
            logit_t = self.og_model(images)

        loss = F.cross_entropy(output, target)
        loss += self.opt.alpha * distill_kl_loss(output, logit_t, self.opt.kd_T)
        
        if self.maximize:
            loss = -loss

        self.top1(output, target)
        return loss


    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        self.maximize=False
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize=True
                time_start = time.process_time()
                self.train_one_epoch(loader=forget_loader)
                self.save_files['train_time_taken'] += time.process_time() - time_start
                self.eval(loader=test_loader)

            self.maximize=False
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
            self.eval(loader=test_loader)
        return
    

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return


class BadT(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        self.og_model.to(self.opt.device)
        self.og_model.eval()
        self.random_model = unlearn_func(model, 'EU')
        self.random_model.eval()
        self.kltemp = 1
    

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        
        full_teacher_logits = self.og_model(images)
        unlearn_teacher_logits = self.random_model(images)
        f_teacher_out = torch.nn.functional.softmax(full_teacher_logits / self.kltemp, dim=1)
        u_teacher_out = torch.nn.functional.softmax(unlearn_teacher_logits / self.kltemp, dim=1)
        labels = torch.unsqueeze(infgt, 1)
        overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
        student_out = F.log_softmax(output /self.kltemp, dim=1)
        loss = F.kl_div(student_out, overall_teacher_out)
        
        self.top1(output, target)
        return loss


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        return 


class SSD(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)


    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()
        self.best_model = ssd_tuning(self.model, forget_loader, self.opt.SSDdampening, self.opt.SSDselectwt, train_loader, self.opt.device)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        return


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.SSDdampening)+'_'+str(self.opt.SSDselectwt)
        return 
    
class SpectralSignature(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super(SpectralSignature, self).__init__(opt, model, prenet)
        # Initialize parameters for spectral signature analysis
        self.spectral_threshold = 1000  # Threshold for identifying significant singular values
        self.contribution_threshold = 10  # Threshold for identifying significant data point contributions

    def forward_pass(self, images, target, infgt):
        # Utilize the forward_pass from ApplyK as a baseline
        output = super(SpectralSignature, self).forward_pass(images, target, infgt)
        return output

    def spectral_analysis(self, activations):
        # Perform SVD on the activations
        activations = activations.float()
        U, S, V = torch.svd(activations, some=True, compute_uv=True)
        # Identify significant singular values (this is highly conceptual and depends on your application)
        significant_svs = S > self.spectral_threshold
        # Calculate the contribution of each data point to the significant singular values
        contributions = torch.matmul(U[:, significant_svs], torch.diag(S[significant_svs]))
        # Identify data points with contributions above a certain threshold
        significant_data_points = contributions.norm(dim=1) > self.contribution_threshold
        return significant_data_points

    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        activations_list = []
        self.model.eval()
        with torch.no_grad():
            for images, target, infgt in train_loader:
                images = images.cuda()
                activations = self.model(images)
                activations_list.append(activations)
        all_activations = torch.cat(activations_list, dim=0)
        significant_data_points = self.spectral_analysis(all_activations)
        filtered_dataset = [
            data for i, data in enumerate(train_loader.dataset)
            if i not in significant_data_points.nonzero()
        ]
        new_train_loader = DataLoader(filtered_dataset, batch_size=train_loader.batch_size, shuffle=False)
        self.model = unlearn_func(self.model, 'EU')
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.scaler = torch.cuda.amp.GradScaler()
        print(f"Retraining model without {len(significant_data_points)} identified targets.")

        # Retrain the model with the filtered dataset
        for epoch in range(self.opt.unlearn_iters):
            self.train_one_epoch(new_train_loader)
        self.eval(test_loader)
        
        print("Unlearning process completed.")


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return

class ActivationClustering(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.task = ClassificationTask()
        self.model = prepare_model(model=model, task=self.task)
        self.pca = PCA(n_components=3)
        self.nb_clusters = 2  
        self.clusterer = KMeans(n_clusters=self.nb_clusters, random_state=0)

    def _get_activations(self, data_loader):
        """
        Get activations from the model for each class separately.
        """
        activations_by_class = {}
        self.model.eval()
        with torch.no_grad():
            for images, labels, _ in data_loader:
                images = images.to(self.opt.device)
                output = self.model(images)
                for img, label in zip(output.detach().cpu(), labels):
                    label = label.item()
                    if label not in activations_by_class:
                        activations_by_class[label] = []
                    activations_by_class[label].append(img.numpy())
        for label in activations_by_class:
            activations_by_class[label] = np.array(activations_by_class[label])
        return activations_by_class

    def _perform_activation_clustering(self, activations_by_class):
        cluster_labels_by_class = {}
        for label, activations in activations_by_class.items():
            activations = self.pca.fit_transform(activations)
            cluster_labels = self.clusterer.fit_predict(activations)
            cluster_labels_by_class[label] = cluster_labels
        return cluster_labels_by_class

    def _identify_unlearning_targets(self, cluster_labels_by_class):
        """
        Identify the samples that should be unlearned based on cluster analysis for each class.
        """
        unlearning_targets = set()
        for label, cluster_labels in cluster_labels_by_class.items():
            counts = np.bincount(cluster_labels)
            target_cluster = np.argmin(counts)
            targets = np.where(cluster_labels == target_cluster)[0]
            unlearning_targets.update(targets)
        return list(unlearning_targets)

    def _filter_loader(self, loader, targets):
        # Assuming loader.dataset is a list or similar; adjust for actual data structure
        filtered_dataset = [data for i, data in enumerate(loader.dataset) if i not in targets]
        return torch.utils.data.DataLoader(filtered_dataset, batch_size=loader.batch_size, shuffle=False)

    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        print("Starting unlearning process...")

        # Get activations from the model 
        activations_by_class = self._get_activations(train_loader)
        
        # Perform activation clustering on these activations
        cluster_labels = self._perform_activation_clustering(activations_by_class)
        
        # Identify unlearning targets based on cluster analysis
        unlearning_targets = self._identify_unlearning_targets(cluster_labels)
        detected_idx_path = '/data/jiawei_li/corrective-unlearning-bench/notebook/harmful_idx.txt'
        with open(detected_idx_path, 'w') as file:
            file.write(', '.join(map(str, sorted(unlearning_targets))))

        print("remove: ", len(unlearning_targets))
        
        # Filter out the unlearning targets from the train_loader
        new_train_loader = self._filter_loader(train_loader, unlearning_targets)
        model = getattr(resnet, self.opt.model)(self.opt.num_classes)
        super().__init__(self.opt, model, None)  
        self.model = prepare_model(model=model, task=self.task)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        # Re-train the model with the filtered training set using existing methods
        while self.curr_step < self.opt.unlearn_iters:
            self.train_one_epoch(new_train_loader)
            self.eval(test_loader)
        
        print("Unlearning process completed.")

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]

class ClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: torch.nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        if len(batch) == 3:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return torch.nn.functional.cross_entropy(logits, sampled_labels.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

class InfluenceFunction(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.task = ClassificationTask()
        self.model = prepare_model(model=self.model, task=self.task)
        self.analyzer = Analyzer(analysis_name="unlearn_analysis", model=self.model, task=self.task)
        self.threshold = 0.0

    def fit_influence_factors(self, train_loader):
        wrapped_train_loader = DataLoader(DataSetWrapper(train_loader.dataset), batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers, pin_memory=True)
        # Fit all EKFAC factors for the given model
        #self.analyzer.fit_all_factors(factors_name="ekfac", dataset=train_dataset)
        factor_strategy = 'ekfac'
        factor_args = FactorArguments(strategy=factor_strategy)
        self.analyzer.fit_all_factors(
            factors_name=factor_strategy,
            dataset=wrapped_train_loader.dataset,
            per_device_batch_size=None,
            factor_args=factor_args,
            overwrite_output_dir=True,
        )

    def compute_influences(self, test_loader, train_loader):
        wrapped_train_loader = DataSetWrapper(CollectedDataset(train_loader))
        wrapped_test_loader = DataSetWrapper(CollectedDataset(test_loader))
        # Compute all pairwise influence scores with the computed factors
        self.analyzer.compute_pairwise_scores(
            scores_name="unlearn_scores",
            factors_name="ekfac",
            query_dataset=wrapped_test_loader.dataset,
            train_dataset=wrapped_train_loader.dataset,
            per_device_query_batch_size=1024,  # Adjust based on your GPU capacity
        )
        scores = self.analyzer.load_pairwise_scores(scores_name="unlearn_scores")
        return scores['all_modules']

    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        # First, fit the influence factors using the training data
        self.fit_influence_factors(train_loader)

        # Compute influences of the test data on the training data
        influences = self.compute_influences(test_loader, train_loader)

        # Identify training samples with the highest influence scores
        harmful_indices = self.identify_harmful(influences)
        print("remove samples: ", len(harmful_indices))

        # Filter out the most harmful training samples
        new_train_loader = self.filter_training_data(train_loader, harmful_indices)
        self.model = unlearn_func(self.model, 'EU')
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.scaler = torch.cuda.amp.GradScaler()

        # Re-train the model with the filtered training set using existing methods
        while self.curr_step < self.opt.unlearn_iters:
            self.train_one_epoch(new_train_loader)
            self.eval(test_loader)

    def identify_harmful(self, influences):
        # Select harmful samples based on influence score threshold
        harmful_scores = influences.sum(axis=0)  # Sum influences per training sample
        harmful_indices = [i for i,e in enumerate(harmful_scores) if harmful_scores[i] < self.threshold]  
        return harmful_indices

    def filter_training_data(self, train_loader, harmful_indices):
        # Filter out harmful data points
        new_dataset = [data for i, data in enumerate(train_loader.dataset) if i not in harmful_indices]
        new_train_loader = DataLoader(new_dataset, batch_size=train_loader.batch_size, shuffle=False)
        return new_train_loader

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return

class FlippingInfluence(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.task = ClassificationTask()
        self.model = prepare_model(model=self.model, task=self.task)
        self.analyzer = Analyzer(analysis_name='unlearn_analysis', model=self.model, task=self.task)

    def fit_influence_factors(self, train_loader):
        # Fit all EKFAC factors for the given model
        collected_train_data = CollectedDataset(train_loader)
        wrapped_train_dataset = DataSetWrapper(collected_train_data)
        self.analyzer.fit_all_factors(
            factors_name="ekfac",
            dataset=wrapped_train_dataset,
            per_device_batch_size=25,
            factor_args=FactorArguments(strategy='ekfac'),
            overwrite_output_dir=True,
        )

    def compute_influences(self, train_loader, deletion_loader, score_name):
        self.analyzer.compute_pairwise_scores(
            scores_name=score_name, 
            factors_name="ekfac",
            train_dataset=train_loader,
            query_dataset=deletion_loader,
            per_device_query_batch_size=25,
            overwrite_output_dir=True
        )
        return self.analyzer.load_pairwise_scores(score_name)

    def flip_images(self, loader):
        flipped_images = []
        targets = []
        for img, target in loader:
            flipped_image = transforms.functional.hflip(img)
            flipped_images.append(flipped_image) 
            targets.append(target)
        flipped_dataset = CustomDataset(flipped_images, targets)  
        return DataLoader(flipped_dataset, batch_size=50, shuffle=False) 

    def detect_poisons(self, n_tolerate, train_loader, deletion_loader, save_dir):
        wrapped_train_dataset = DataSetWrapper(CollectedDataset(train_loader))
        wrapped_deletion_dataset = DataSetWrapper(CollectedDataset(deletion_loader))
        
        # Step 1: Calculate initial influence scores
        original_scores_name = "old_scores"
        original_scores = self.compute_influences(wrapped_train_dataset, wrapped_deletion_dataset, original_scores_name)

        # Step 2: Flip the images in the deletion set
        flipped_loader = self.flip_images(wrapped_deletion_dataset)
        wrapped_flipped_dataset = DataSetWrapper(CollectedDataset(flipped_loader))

        # Step 3: Recalculate influence scores with flipped images
        flipped_scores_name = "new_scores"
        flipped_scores = self.compute_influences(wrapped_train_dataset, wrapped_flipped_dataset, flipped_scores_name)
        
        original_scores = original_scores['all_modules']
        flipped_scores = flipped_scores['all_modules']

        # Step 4: Calculate delta matrix
        delta_scores = flipped_scores - original_scores
        # print(type(delta_scores), delta_scores.size())
        # print(delta_scores)

        # Get the labels for the train and deletion datasets
        train_labels = torch.tensor([sample[1] for sample in wrapped_train_dataset])
        deletion_labels = torch.tensor([sample[1] for sample in wrapped_deletion_dataset])

        # Ensure the labels are the same size as delta_scores
        print("delta_scores.size:", delta_scores.size(0))
        print("deletion_labels.size:", deletion_labels.size(0))
        print("train_labels.size:", train_labels.size(0))
        assert delta_scores.size(0) == deletion_labels.size(0) and delta_scores.size(1) == train_labels.size(0)

        # Calculate the number of positive scores for each row in delta_scores_sub
        # but only count the scores where the labels match
        #label_match_mask = deletion_labels.unsqueeze(1) == train_labels.unsqueeze(0)
        positive_delta_mask = delta_scores >= 0
        #combined_mask = label_match_mask & positive_delta_mask
        #positive_counts = combined_mask.sum(dim=0) 
        positive_counts = positive_delta_mask.sum(dim=0)

        # Determine which cols have positive_counts less than self.n_tolerate
        valid_indices = torch.where(positive_counts <= n_tolerate)[0]

        # Convert valid_indices to a set for consistency with the original code
        poison_indices = set(valid_indices.tolist())
        
        print(f"{len(poison_indices)} poisons detected...")
        #print(f"poison_indices: {poison_indices}")
        # save detected indices (change to yours dir)
        detected_idx_path = '/data/jiawei_li/corrective-unlearning-bench/notebook/harmful_idx.txt'
        with open(detected_idx_path, 'w') as file:
            file.write(', '.join(map(str, sorted(poison_indices))))

        return poison_indices

    def filter_training_data(self, train_loader, harmful_indices):
        # Filter out harmful data points
        new_dataset = [data for i, data in enumerate(train_loader.dataset) if i not in harmful_indices]
        print("new dataset", len(new_dataset), "train loader", len(train_loader.dataset))
        new_train_loader = DataLoader(new_dataset, batch_size=train_loader.batch_size, shuffle=True)
        return new_train_loader

    def unlearn(self, n_tolerate, train_loader, test_loader, deletion_loader, deletion_idx, save_dir):
        self.fit_influence_factors(train_loader)
        harmful_indices = self.detect_poisons(n_tolerate, train_loader, deletion_loader, save_dir)
        harmful_indices.update(deletion_idx.tolist())
        # print(f"remove samples: ({len(harmful_indices)})")
        new_train_loader = self.filter_training_data(train_loader, harmful_indices)
        # retrain from scratch 
        model = getattr(resnet, self.opt.model)(self.opt.num_classes)
        super().__init__(self.opt, model, None)  
        self.model = prepare_model(model=model, task=self.task)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        while self.curr_step < self.opt.unlearn_iters:
            self.train_one_epoch(new_train_loader)
            self.eval(test_loader)

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return

class SwappingInfluence(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.task = ClassificationTask()
        self.model = prepare_model(model=self.model, task=self.task)
        self.analyzer = Analyzer(analysis_name='unlearn_analysis', model=self.model, task=self.task)
        self.threshold = 0.0
        self.num_topk = 500

    def fit_influence_factors(self, train_loader):
        collected_train_data = CollectedDataset(train_loader)
        wrapped_train_dataset = DataSetWrapper(collected_train_data)
        self.analyzer.fit_all_factors(
            factors_name="ekfac",
            dataset=wrapped_train_dataset,
            per_device_batch_size=250,
            factor_args=FactorArguments(strategy='ekfac'),
            overwrite_output_dir=True,
        )

    def compute_influences(self, train_loader, query_loader, score_name):
        self.analyzer.compute_pairwise_scores(
            scores_name=score_name,
            factors_name="ekfac",
            train_dataset=train_loader,
            query_dataset=query_loader,
            per_device_query_batch_size=100,
            overwrite_output_dir=True
        )
        return self.analyzer.load_pairwise_scores(score_name)
    
    def flip_images(self, loader):
        flipped_images = []
        targets = []
        for img, target in loader:
            flipped_image = transforms.functional.hflip(img)
            flipped_images.append(flipped_image) 
            targets.append(target)
        flipped_dataset = CustomDataset(flipped_images, targets)  
        return DataLoader(flipped_dataset, batch_size=50, shuffle=False) 

    def detect_label_swap(self, train_loader, delete_idx, threshold=0, num_topk=500):
        collected_train_data = CollectedDataset(train_loader)
        wrapped_train_dataset = DataSetWrapper(collected_train_data)

        # Focus on examples with the same label Y
        deletion_samples = [train_loader.dataset[idx] for idx in delete_idx]
        labels = [train_loader.dataset[idx][1] for idx in delete_idx]
        unique_labels = set(labels)
        print("unique labels:", unique_labels)
        detected_poisons = set()

        for Y in unique_labels:
            # Step 1: Calculate influence scores on Q (influence on Y examples in deletion set)
            query_dataset_Q = CustomDataset(
                images=[example[0] for example in deletion_samples if example[1] == Y],
                targets=[example[1] for example in deletion_samples if example[1] == Y]
            )
            print(f"Length of images: {len(query_dataset_Q.images)}")
            print(f"Length of targets: {len(query_dataset_Q.targets)}")
            old_scores = self.compute_influences(wrapped_train_dataset, query_dataset_Q, "old_scores")['all_modules']

            # Step 2: Calculate influence scores on Q' (influence on Y examples in training set)
            #query_dataset_Q_prime_loader = self.flip_images(query_dataset_Q)
            #query_dataset_Q_prime = CollectedDataset(query_dataset_Q_prime_loader)

            query_dataset_Q_prime = CustomDataset(
                images=[example[0] for example in wrapped_train_dataset if example[1] == Y],
                targets=[example[1] for example in wrapped_train_dataset if example[1] == Y]
            )
            #print(f"Length of images: {len(query_dataset_Q_prime.images)}")
            #print(f"Length of targets: {len(query_dataset_Q_prime.targets)}")
            new_scores = self.compute_influences(wrapped_train_dataset, query_dataset_Q_prime, "new_scores")['all_modules']
            print("size old scores:", old_scores.size(), "size new scores:", new_scores.size())

            # Step 3: Calculate delta scores
            new_scores = new_scores.mean(dim=0)
            old_scores = old_scores.mean(dim=0)
            delta_scores = new_scores - old_scores
            delta_scores_np = delta_scores.cpu().numpy()

            # Step 4: Identify detected poisons
            #np.savez("delta_scores.npz", delta_scores=delta_scores)
            #negative_counts = (delta_scores < 0).sum(dim=0) 
            #print("negative counts size:", negative_counts.size())
            #negative_counts_np = negative_counts.cpu().numpy()
            #detected_poison = [i for i, score in enumerate(delta_scores_np) if score > threshold]
            sorted_indexes = np.argsort(delta_scores_np)
            detected_poison = sorted_indexes[:num_topk].tolist()
            detected_poisons.update(detected_poison)

        return detected_poisons

    def unlearn(self, train_loader, test_loader, delete_idx, threshold=0, num_topk=500):
        # Fit influence factors
        self.fit_influence_factors(train_loader)
        harmful_indices = self.detect_label_swap(train_loader, delete_idx,threshold=threshold, num_topk=num_topk)
        print(f"Detected {len(harmful_indices)} poisons ...")
        harmful_indices.update(delete_idx.tolist())
        manip_idx_path = '/data/jiawei_li/corrective-unlearning-bench/notebook/harmful_idx.txt'
        with open(manip_idx_path, 'w') as file:
            file.write(', '.join(map(str, sorted(harmful_indices))))

        # Filter out the harmful data points
        new_train_loader = self.filter_training_data(train_loader, harmful_indices)  
        
        model = getattr(resnet, self.opt.model)(self.opt.num_classes)
        super().__init__(self.opt, model, None)  
        self.model = prepare_model(model=model, task=self.task)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        
        # Retrain the model with the filtered training set
        while self.curr_step < self.opt.unlearn_iters:
            self.train_one_epoch(new_train_loader)
            self.eval(test_loader)

    def filter_training_data(self, train_loader, harmful_indices):
        new_dataset = [data for i, data in enumerate(train_loader.dataset) if i not in harmful_indices]
        print(f"Filtered dataset: {len(new_dataset)} samples remaining.")
        new_train_loader = DataLoader(new_dataset, batch_size=train_loader.batch_size, shuffle=True)
        return new_train_loader

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix + '/' + str(self.opt.deletion_size) + '_' + self.opt.unlearn_method + '_' + self.opt.exp_name
        self.unlearn_file_prefix += '_' + str(self.opt.train_iters) + '_' + str(self.opt.k)
        self.unlearn_file_prefix += '_' + str(self.opt.kd_T) + '_' + str(self.opt.alpha) + '_' + str(self.opt.msteps)
        return