import torch, torchmetrics, tqdm, copy, time
from utils import LinearLR, unlearn_func, ssd_tuning, distill_kl_loss, compute_accuracy
from torch.cuda.amp import autocast
import numpy as np
from torch.cuda.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools
from sklearn.cluster import KMeans
import torch.nn as nn


class Naive():
    def __init__(self, opt, model, prenet=None):
        self.opt = opt
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.set_model(model, prenet)
        self.save_files = {'train_top1':[], 'val_top1':[], 'train_time_taken':0}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.max_lr, momentum=0.9, weight_decay=self.opt.wd)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.train_iters*1.25, warmup_epochs=self.opt.train_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.opt.num_classes).cuda()
        self.scaler = GradScaler()


    def set_model(self, model, prenet=None):
        self.prenet = None
        self.model = model
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
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
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
        # Example of unlearning based on spectral analysis (this is a simplified example)
        self.model.train()
        for images, target, infgt in train_loader:
            images, target = images.cuda(), target.cuda()
            with autocast():
                self.optimizer.zero_grad()
                activations = self.model(images)  # Obtain activations
                significant_data_points = self.spectral_analysis(activations)
                # Filter out significant data points based on spectral analysis for this training step
                filtered_images = images[~significant_data_points]
                filtered_targets = target[~significant_data_points]
                # Perform training step with filtered data
                if len(filtered_images) > 0:  # Check if there are any data points left after filtering
                    output = self.model(filtered_images)
                    loss = F.cross_entropy(output, filtered_targets)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
        super(SpectralSignature, self).eval(test_loader)

    def get_save_prefix(self):
        # Customize the save prefix to include details specific to SpectralSignature
        base_prefix = super(SpectralSignature, self).get_save_prefix()  # Get the base prefix from ApplyK
        spectral_prefix = "SpectralSig_threshold{}_contrib{}".format(self.spectral_threshold, self.contribution_threshold)
        self.unlearn_file_prefix = "{}/{}".format(base_prefix, spectral_prefix)
        
        return

class ActivationClustering(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.nb_clusters = 2  
        self.clusterer = KMeans(n_clusters=self.nb_clusters, random_state=0)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.unlearn_lr)
        self.epoch_loss = 0.0

    def _get_activations(self, data_loader):
        activations = []
        self.model.eval()
        with torch.no_grad():
            for images, _, _ in data_loader:
                images = images.to(self.opt.device)
                output = self.model(images)
                activations.append(output.detach().cpu())
        activations = torch.cat(activations, dim=0)
        return activations.numpy()

    def _perform_activation_clustering(self, activations):
        cluster_labels = self.clusterer.fit_predict(activations)
        return cluster_labels

    def _identify_unlearning_targets(self, cluster_labels):
        counts = np.bincount(cluster_labels)
        target_cluster = np.argmin(counts)
        return np.where(cluster_labels == target_cluster)[0]

    def _filter_loader(self, loader, targets):
        # Assuming loader.dataset is a list or similar; adjust for actual data structure
        filtered_dataset = [data for i, data in enumerate(loader.dataset) if i not in targets]
        return torch.utils.data.DataLoader(filtered_dataset, batch_size=loader.batch_size, shuffle=True)

    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        print("Starting unlearning process...")

        # Get activations from the model for the forget_loader dataset
        activations = self._get_activations(train_loader)
        
        # Perform activation clustering on these activations
        cluster_labels = self._perform_activation_clustering(activations)
        
        # Identify unlearning targets based on cluster analysis
        unlearning_targets = self._identify_unlearning_targets(cluster_labels)
        
        # Filter out the unlearning targets from the train_loader
        new_train_loader = self._filter_loader(train_loader, unlearning_targets)

        print(f"Retraining model without {len(unlearning_targets)} identified targets.")
        
        for epoch in range(self.opt.unlearn_iters):  
            self.train_one_epoch(new_train_loader)
        
        self.eval(test_loader)
        
        print("Unlearning process completed.")

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        for images, targets, _ in loader:
            images, targets = images.to(self.opt.device), targets.to(self.opt.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)  # Assuming self.criterion is defined
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item() * images.size(0)
        self.epoch_loss = running_loss / len(loader.dataset)
        print(f'Training Loss: {self.epoch_loss:.6f}')

    def get_save_prefix(self):
        base_prefix = super(ActivationClustering, self).get_save_prefix()  # Get the base prefix from ApplyK
        spectral_prefix = "ActivationClustering_nbClusters{}".format(self.nb_clusters)
        self.unlearn_file_prefix = "{}/{}".format(base_prefix, spectral_prefix)
        return