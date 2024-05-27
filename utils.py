import torch
import os
import joblib
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tqdm import tqdm
import numpy as np
import math
import threading
import queue as Queue
criterion_md = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()

def PIT_calculation(y_pred,y_true):
    mu = y_pred[:,:5] + 1e-6
    sigma = y_pred[:,5:10] + 1e-6
    weight = y_pred[:,10:15] + 1e-6

    # normalize the weight
    weight = weight / torch.sum(weight,dim=1,keepdims=True)

    # Create normal distribution.
    dist = torch.distributions.normal.Normal(mu,sigma)

    cdf = dist.cdf(y_true)

    return torch.sum(weight*cdf,dim=1,keepdim=True)

#CRPS loss
def CRPS_loss(y_pred,y_true):
    mu = y_pred[:,:5] + 1e-6
    sigma = y_pred[:,5:10] + 1e-6
    weight = y_pred[:,10:15] + 1e-6
    
    # normalize the weight
    weight = weight / torch.sum(weight,dim=1,keepdims=True)

    # Create normal distribution.
    dist = torch.distributions.normal.Normal(mu,sigma)
    standard_normal = torch.distributions.normal.Normal(0, 1)
    
    '''
    Analytical solution of the CRPS for the normal distribution.   
    Detailed formula introduced in http://dx.doi.org/10.1175/MWR2906.1.
    '''

    omiga = (y_true - mu)/sigma   
    cdfNormal = standard_normal.cdf(omiga)
    pdfNormal = torch.exp(standard_normal.log_prob(omiga))
    
    CRPS = weight * (sigma * (omiga*(2*cdfNormal-1) + 2*pdfNormal - math.pi ** (-0.5)))
    
    #return torch.sum(CRPS,dim=1).mean()
    return torch.sum(CRPS,dim=1,keepdim=True)

#mixture density network loss
def mdn_loss(x, target):
    sigma = x[:,:5] + 1e-6
    mu = x[:,5:] + 1e-6
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = m.log_prob(target)
    loss = torch.exp(loss)
    loss = torch.sum(loss, dim=1)
    loss = -torch.log(loss + 1e-6)
    return torch.mean(loss)

def quasar_contrast_loss(photo_feature, image_feature, labels, t=0.21, gamma=0.13):
    photo_feature = F.normalize(photo_feature, dim=1)
    image_feature = F.normalize(image_feature, dim=1)
    # matrix similarity: NxN
    sim_photo2image = torch.matmul(photo_feature, image_feature.T) / t
    sim_photo2photo = torch.matmul(photo_feature, photo_feature.T) / t
    sim_image2image = torch.matmul(image_feature, image_feature.T) / t

    #sim_photo2image: Calculate the number of common classes between nXn.
    #label_sim: What is actually computed is the number of identical classes within the same batch size.
    label_sim = torch.matmul(labels, labels.T).clamp(max=1.0)
    #label_sim = label_sim ** 0.5
    pro_inter = label_sim / label_sim.sum(1, keepdim=True).clamp(min=1e-6)
    label_sim_intra = (label_sim - torch.eye(label_sim.shape[0]).cuda()).clamp(min=0)
    pro_intra = label_sim_intra / label_sim_intra.sum(1, keepdim=True).clamp(min=1e-6)

    # logits: NxN
    logits_photo2image = sim_photo2image - torch.log(torch.exp(1.06 * sim_photo2image).sum(1, keepdim=True))
    logits_image2photo = sim_photo2image.T - torch.log(torch.exp(1.06 * sim_photo2image.T).sum(1, keepdim=True))
    logits_photo2photo = sim_photo2photo - torch.log(torch.exp(1.06 * sim_photo2photo).sum(1, keepdim=True))
    logits_image2image = sim_image2image - torch.log(torch.exp(1.06 * sim_image2image).sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos_photo2image = (pro_inter * logits_photo2image).sum(1)
    mean_log_prob_pos_image2photo = (pro_inter * logits_image2photo).sum(1)
    mean_log_prob_pos_photo2photo = (pro_intra * logits_photo2photo).sum(1)
    mean_log_prob_pos_image2image = (pro_intra * logits_image2image).sum(1)

    # supervised cross-modal contrastive loss
    loss = - mean_log_prob_pos_photo2image.mean() - mean_log_prob_pos_image2photo.mean() \
           - gamma * (mean_log_prob_pos_photo2photo.mean() + mean_log_prob_pos_image2image.mean())

    return loss


def gan_loss(view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2\
             ,label,contrast_class,z1,z2,z3,z4,criterion):
    
    """
    The loss function utilized for the training set.
    """
    
    bs = view1_modal_view1.size()[0]
    photo_md = torch.ones(bs, dtype=torch.long).cuda()
    img_md = torch.zeros(bs, dtype=torch.long).cuda()
     
    c1 = criterion_md(view1_modal_view1, photo_md)
    c2 = criterion_md(view2_modal_view1, img_md)
    c3 = criterion_md(view1_modal_view2, photo_md)
    c4 = criterion_md(view2_modal_view2, img_md)
    
    
    z1_loss = criterion(z1,label)
    z2_loss = criterion(z2,label)
    z3_loss = criterion(z3,label)
    z4_loss = criterion(z4,label)
   
    contrast_loss = quasar_contrast_loss(view1_modal_view1,view2_modal_view2,contrast_class)
    
    return 1.5*(z1_loss)+0.3*(c1+c2+c3+c4)+0.8*contrast_loss,c1.detach(),c2.detach(),c3.detach(),c4.detach(),contrast_loss.detach(),z2_loss.detach(),z3_loss.detach(),z4_loss.detach(),z1_loss.detach()

def gan1_loss(view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2\
             ,label,contrast_class,z1,z2,z3,z4,criterion):
    
    """
    The loss function employed for the validation set.
    """
    
    bs = view1_modal_view1.size()[0]
    photo_md = torch.ones(bs, dtype=torch.long).cuda()
    img_md = torch.zeros(bs, dtype=torch.long).cuda()

    c1 = criterion_md(view1_modal_view1, photo_md)
    c2 = criterion_md(view2_modal_view1, img_md)
    c3 = criterion_md(view1_modal_view2, photo_md)
    c4 = criterion_md(view2_modal_view2, img_md)

    z1_loss = criterion(z1,label)
    z2_loss = criterion(z2,label)
    z3_loss = criterion(z3,label)
    z4_loss = criterion(z4,label)

    contrast_loss = quasar_contrast_loss(view1_modal_view1,view2_modal_view2,contrast_class)
    
    return z1_loss.detach(),c1.detach(),c2.detach(),c3.detach(),c4.detach(),contrast_loss.detach(),z1_loss.detach(),z2_loss.detach(),z3_loss.detach(),z4_loss.detach()   
    
def train_one_modal(train_loader,model,regression_model,optimizer,criterion,scaler,modal):
    model.train()
    regression_model.train()
    mean_loss = torch.zeros(1).cuda()
    train_loader=tqdm(train_loader)
    for i, (x1,x2,image,labels,con_labels) in enumerate(train_loader):
        x1=x1.cuda()
        x2=x2.cuda()
        # Determine whether the tensor contains null values.
        contains_nan = torch.isnan(x1).any().item()
        image=image.cuda()
        labels=labels.cuda()
        with torch.cuda.amp.autocast():
            if modal=='photo':
               photo_feature = model(x1,x2)
               z=regression_model(photo_feature)
            elif modal=='img':
               img_feature =  model(image)
               z=regression_model(img_feature)
            loss = criterion(z,labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        mean_loss = (mean_loss * i + loss.detach().item()) / (i + 1)
        train_loader.desc = "[epoch {}] mean loss {}".format(i, round(mean_loss[0].item(), 3))
    return mean_loss    

@torch.no_grad()
def evaluate_one_modal(test_loader,model,regression_model,criterion,modal):
    model.eval()
    #regression_model.eval()
    mean_loss = torch.zeros(1).cuda()
    test_loader = tqdm(test_loader)
    for i, (x1,x2,image,labels,con_labels) in enumerate(test_loader):
        x1=x1.cuda()
        x2=x2.cuda()
        image=image.cuda()
        labels=labels.cuda()
        if modal=='photo':
            photo_feature = model(x1,x2)
            z=regression_model(photo_feature)
        elif modal=='img':
            img_feature =  model(image)
            z=regression_model(img_feature)
        loss = criterion(z,labels)
        mean_loss = (mean_loss * i + loss.detach().item()) / (i + 1)
        test_loader.desc = "[epoch {}] mean mse {}".format(i, round(mean_loss.item(), 3))

    return mean_loss

def train_one_epoch(train_loader,model,regression_model,optimizer,criterion,scaler,use_mix=False):
    model.train()
    regression_model.train()
    mean_loss = torch.zeros(10).cuda()
    train_loader = tqdm(train_loader)
    for i, (x1,x2,image,labels,con_labels) in enumerate(train_loader):
        x1=x1.cuda()
        x2=x2.cuda()
        # Determine whether the tensor contains null values.
        contains_nan = torch.isnan(x1).any().item()

        if contains_nan:
            print("The tensor contains NaN values.")
        
        image=image.cuda()
        labels=labels.cuda()
        con_labels=con_labels.cuda()
        
        if use_mix==True:
            with torch.cuda.amp.autocast():
                img_feature,photo_feature,img2photo_feature,photo2img_feature,img2img_judge,photo2img_judge,img2photo_judge,\
                photo2photo_judge = model(x1,x2,image)
            
                z1 = regression_model(torch.cat((photo_feature,img_feature),dim=1))
                z2 = regression_model(torch.cat((photo_feature,photo2img_feature),dim=1))
                z3 = regression_model(torch.cat((img2photo_feature,img_feature),dim=1))
                z4 = regression_model(torch.cat((img2photo_feature,photo2img_feature),dim=1))
        
                loss = gan_loss(photo2photo_judge,img2photo_judge,photo2img_judge,img2img_judge,labels,\
                      con_labels,z1,z2,z3,z4,criterion)

                optimizer.zero_grad()
                scaler.scale(loss[0]).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                scaler.unscale_(optimizer)
        
            scaler.step(optimizer)
            scaler.update()

        else:
            optimizer.zero_grad()
            img_feature,photo_feature,img2photo_feature,photo2img_feature,img2img_judge,photo2img_judge,img2photo_judge,\
                photo2photo_judge = model(x1,x2,image)

            z1 = regression_model(torch.cat((photo_feature,img_feature),dim=1))
            z2 = regression_model(torch.cat((photo_feature,photo2img_feature),dim=1))
            z3 = regression_model(torch.cat((img2photo_feature,img_feature),dim=1))
            z4 = regression_model(torch.cat((img2photo_feature,photo2img_feature),dim=1))

            loss = gan_loss(photo2photo_judge,img2photo_judge,photo2img_judge,img2img_judge,labels,\
                      con_labels,z1,z2,z3,z4,criterion)

            loss[0].backward() 
            optimizer.step()

        for j in range(10):
            mean_loss[j] = (mean_loss[j] * i + loss[j].detach().item()) / (i + 1)
        train_loader.desc = "[epoch {}] mean loss {}".format(i, round(mean_loss[0].item(), 3))
    
    return mean_loss
        
@torch.no_grad()
def evaluate(test_loader,model,regression_model,criterion):
    model.eval()
    #regression_model.eval()
    mean_loss = torch.zeros(10).cuda()
    test_loader = tqdm(test_loader)
    for i, (x1,x2,image,labels,con_labels) in enumerate(test_loader):
        x1=x1.cuda()
        x2=x2.cuda()
        image=image.cuda()
        labels=labels.cuda()
        con_labels=con_labels.cuda()
        img_feature,photo_feature,img2photo_feature,photo2img_feature,img2img_judge,photo2img_judge,img2photo_judge,\
            photo2photo_judge = model(x1,x2,image)
        
        z1 = regression_model(torch.cat((photo_feature,img_feature),dim=1))
        z2 = regression_model(torch.cat((photo_feature,photo2img_feature),dim=1))
        z3 = regression_model(torch.cat((img2photo_feature,img_feature),dim=1))
        z4 = regression_model(torch.cat((img2photo_feature,photo2img_feature),dim=1))

        loss = gan1_loss(photo2photo_judge,img2photo_judge,photo2img_judge,img2img_judge,labels,con_labels,z1,z2,z3,z4,criterion)
        for j in range(10):
            mean_loss[j] = (mean_loss[j] * i + loss[j].detach().item()) / (i + 1)
        test_loader.desc = "[epoch {}] mean mse {}".format(i, round(mean_loss[0].item(), 3))
    
    return mean_loss

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


# decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs), max_prefetch=self.max_prefetch)

        return bg_generator
    
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

class PhotoDataset(Dataset):
    def __init__(self,x1,x2,y,con_labels,prepath,extension,types):
        # modal 0:means photo 1:means image 2:both of them
        self.transform = ToTensor()
        self.x1 = self.transform(x1).permute(1,2,0).to(torch.float32)
        self.x2 = self.transform(x2).permute(1,2,0).to(torch.float32).squeeze(-1)
        self.y  = torch.tensor(y).to(torch.float32).unsqueeze(-1)
        self.prepath = prepath
        self.con_labels = F.one_hot(torch.tensor(con_labels),12).to(torch.float32)
        #'objID','z','zErr'
        self.extension = extension
        self.types = types
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        if self.types=='SDSS':
            path=os.sep.join((self.prepath,'_'.join((str(self.extension.iloc[idx,0]),str(self.extension.iloc[idx,1])
                                                 ,str(self.extension.iloc[idx,2])))+'.mat'))
        elif self.types=='WISE':
            path=os.sep.join((self.prepath,'-'.join((str(self.extension.iloc[idx,0]),str(self.extension.iloc[idx,1])
                                                 ,str(self.extension.iloc[idx,2])))+'.mat'))
        image=joblib.load(path)
        image=torch.tensor(image).float()
        image = torch.where(
        torch.isnan(image), 
        torch.full_like(image, 0), image)
        return self.x1[idx,:,:],self.x2[idx,:],image,self.y[idx],self.con_labels[idx]

def split_dataset(dataset, test_size=0.2, val_size=0.2, random_state=42,split_type='test',factor=1/1000):
    # generate the index of the dataset
    idx = np.arange(len(dataset))
    y = np.floor((dataset.y*factor))

    # train_test_split to split the datasets into train and test dataset
    train_idx, test_idx, y_train, y_test = train_test_split(idx, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    if split_type=='val':
        '''
        Using the StratifiedShuffleSplit object, we perform stratified random splitting on the training set, dividing it into a training set and a validation set.
        '''
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
        train_idx, val_idx = next(sss.split(train_idx, y_train))

        # Obtain the training set, test set, and validation set based on the indices.
        train_data = torch.utils.data.Subset(dataset, train_idx)
        test_data = torch.utils.data.Subset(dataset, test_idx)
        val_data = torch.utils.data.Subset(dataset, val_idx)

        return train_data, val_data, test_data

    elif split_type=='test':
        # Obtain the training set and test set based on the indices.
        train_data = torch.utils.data.Subset(dataset, train_idx)
        test_data = torch.utils.data.Subset(dataset, test_idx)

        return train_data,test_data

def split_dataset1(dataset, test_size=0.2, val_size=0.2, random_state=42,split_type='test',factor=1/1000):
    # generate the index of the dataset
    idx = np.arange(len(dataset))
    y = np.floor((dataset.y))

    # train_test_split to split the datasets into train and test dataset
    train_idx, test_idx, y_train, y_test = train_test_split(idx, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    if split_type=='val':
        # Utilize the StratifiedShuffleSplit object to perform stratified random splitting on the training set, dividing it into a training set and a validation set.
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
        train_idx, val_idx = next(sss.split(train_idx, y_train))

        # Retrieve the training set, test set, and validation set based on the indices.
        train_data = torch.utils.data.Subset(dataset, train_idx)
        test_data = torch.utils.data.Subset(dataset, test_idx)
        val_data = torch.utils.data.Subset(dataset, val_idx)

        return train_data, val_data, test_data

    elif split_type=='test':
        # Retrieve the training set, test set, and validation set based on the indices.
        train_data = torch.utils.data.Subset(dataset, train_idx)
        test_data = torch.utils.data.Subset(dataset, test_idx)

        return train_data,test_data
