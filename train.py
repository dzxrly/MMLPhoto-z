import faulthandler
faulthandler.enable()
import pandas as pd
import numpy as np
import threading
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from einops import rearrange
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor
from torch import optim
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split,StratifiedKFold
import joblib
import os
import queue as Queue
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from pytorchtools import EarlyStopping
from sklearn.preprocessing import StandardScaler
from utils import PhotoDataset,split_dataset,DataLoaderX,train_one_epoch,evaluate,split_dataset1,train_one_modal,evaluate_one_modal,mdn_loss,CRPS_loss
from model import RegressionClassifier,ContrastNN,SDSSNetwork,SDSSWISENetwork,SKYNetwork,SKYNetworkBand,SDSSPhotoEncoder,SDSSImgEncoder,WiseImgEncoder,WisePhotoEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from SkyMapperDataset import SkyMapperDataset
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def normalized(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return (arr-mean)/std

def read_MN():
    #b=pd.read_csv('MN.csv')
    b=pd.read_csv('./EXTEND.csv')

    b['u-g']=b['dered_u']-b['dered_g']
    b['g-r']=b['dered_g']-b['dered_r']
    b['r-i']=b['dered_r']-b['dered_i']
    b['i-z']=b['dered_i']-b['dered_z']

    extension=b[['bestObjID','z','zErr']]
    y=b['z']
    y2=b[['extinction_u','extinction_g','extinction_r','extinction_i','extinction_z']]

    x1=b[['dered_u','dered_g','dered_r','dered_i','dered_z']]
    x2=b[['u-g','g-r','r-i','i-z']]
    con_labels=b['contrast_class']
    return x1,x2,y,con_labels,extension

def read_YAO():
    #b=pd.read_csv('YAONEW.csv')
    b=pd.read_csv('UNIFY.csv')

    b['u-g']=b['UMAG']-b['GMAG']
    b['g-r']=b['GMAG']-b['RMAG']
    b['r-i']=b['RMAG']-b['IMAG']
    b['i-z']=b['IMAG']-b['ZMAG']
    b['w1-z']=b['ZMAG']-b['W1MAG']
    b['w2-w1']=b['W2MAG']-b['W1MAG']
    b['w3-w2']=b['W3MAG']-b['W2MAG']
    b['w4-w3']=b['W4MAG']-b['W3MAG']
    
    b['SDSS_NAME']=b['SDSS_NAME'].apply(lambda x:x.replace('\'','').replace('b',''))
    extension=b[['SDSS_NAME','Z','COADD_ID']]
    y=b['Z']

    x1=b[['UMAG','GMAG','RMAG','IMAG','ZMAG','W1MAG','W2MAG','W3MAG','W4MAG']]
    x2=b[['u-g','g-r','r-i','i-z','w1-z','w2-w1','w3-w2','w4-w3']]
    con_labels=b['contrast_class']
    return x1,x2,y,con_labels,extension

def read_SKY(task):
    skys=pd.read_csv('SKYSRIZ_NEW.csv')
    skys['u-v']=skys['UPSF']-skys['VPSF']
    skys['v-g']=skys['VPSF']-skys['GPSF']
    skys['g-r']=skys['GPSF']-skys['RPSF']
    skys['r-i']=skys['RPSF']-skys['IPSF']
    skys['i-z']=skys['IPSF']-skys['ZPSF']
    skys['z-w1']=skys['ZPSF']-skys['W1MAG']
    skys['w1-w2']=skys['W1MAG']-skys['W2MAG']
    skys['w2-w3']=skys['W2MAG']-skys['W3MAG']
    skys['w3-w4']=skys['W3MAG']-skys['W4MAG']
   
    paths=skys['image_name_y']
    x1=skys[['UPSF', 'VPSF', 'GPSF', 'RPSF','IPSF', 'ZPSF', 'W1MAG', 'W2MAG', 'W3MAG', 'W4MAG']]
    x2=skys[['u-v','v-g','g-r','r-i','i-z','z-w1','w1-w2','w2-w3','w3-w4']]
    
    if task=='CLASSIFICATION':
       y=skys['CLASS']
    elif task=='ESTIMATION':
       y=skys['Z']
    
    extension=skys[['OBSID','Z']]
    con_labels=skys['contrast_class']
    
    
    return paths,x1,x2,y,con_labels,extension

def main(args):
    mode=args.mode
    task=args.task
    band=args.bands
    modal=args.modal
    
    num_epochs=200
    if mode!='SKY':
        if mode=='SDSS':
            x1,x2,y,con_labels,extension=read_MN()
            image_path='./big_image/'
            types='SDSS'
            s1 = StandardScaler()
            s1.fit_transform(x1.to_numpy())
            s2 = StandardScaler()
            s2.fit_transform(x2.to_numpy())
            x1=s1.transform(x1.to_numpy())
            x2=s2.transform(x2.to_numpy())

        elif mode=='WISE':
            x1,x2,y,con_labels,extension=read_YAO()
            image_path='./YAOIMAGE/'
            types='WISE'
            s1 = StandardScaler()
            s1.fit_transform(x1.to_numpy())
            s2 = StandardScaler()
            s2.fit_transform(x2.to_numpy())
            x1=s1.transform(x1.to_numpy())
            x2=s2.transform(x2.to_numpy())
    else:
        paths,x1,x2,y,con_labels,extension = read_SKY(task)
        s1 = StandardScaler()
        s1.fit_transform(x1.to_numpy())
        s2 = StandardScaler()
        s2.fit_transform(x2.to_numpy())
        x1=s1.transform(x1.to_numpy())
        x2=s2.transform(x2.to_numpy())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = ToTensor()
    
    batch_size = 1024
    
    if mode!='SKY':    
        photoDataset=PhotoDataset(x1,x2,y,con_labels,image_path,extension,types=types)
    else:
        photoDataset=SkyMapperDataset(paths,x1,x2,y,extension,con_labels,types=task,bands=band)
    
    trainDateset,valDataset=split_dataset(photoDataset)

    # load the datasets
    train_loader = DataLoaderX(trainDateset, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(valDataset, batch_size=batch_size, shuffle=True,pin_memory=True)  
    
    
    lrs=[0.04,8*1e-3,5*1e-3,7*1e-4,2*1e-4,8*1e-5,4*1e-5,8*1e-6,2*1e-6,8*1e-7]

    # Initialize the model, loss function, and optimizer.
    for idx,lr in enumerate(lrs):
        writer = SummaryWriter('unify1/experiment_{}'.format(str(lr))) 
        if mode=='SDSS':
            if modal=='photo':
                model = SDSSPhotoEncoder()
            elif modal=='img':
                model = SDSSImgEncoder()
            else:
                model = SDSSNetwork()

        elif mode=='WISE':
            if modal=='photo':
                model = WisePhotoEncoder()
            elif modal=='img':
                model = WiseImgEncoder()
            else: 
                model = SDSSWISENetwork() 

        elif mode=='SKY':
            model = SKYNetworkBand(10-band)

        model = model.cuda()
        criterion = nn.MSELoss()

        if mode!='SKY':
            if modal!='all':
                redshfit_regression = RegressionClassifier(256,num_classes=15)
            else:
                redshfit_regression = RegressionClassifier(512,num_classes=15)

            redshfit_regression = redshfit_regression.cuda()
            #criterion = nn.MSELoss()
            criterion = CRPS_loss
        else:
            if task=='CLASSIFICATION':
                redshfit_regression = RegressionClassifier(512,num_classes=3)
                redshfit_regression = redshfit_regression.cuda()
                criterion = nn.CrossEntropyLoss()
            elif task=='ESTIMATION':
                redshfit_regression = RegressionClassifier(512,num_classes=15)
                redshfit_regression = redshfit_regression.cuda()
                #criterion = nn.MSELoss()
                #criterion = nn.L1Loss()
                criterion = CRPS_loss
        
        # choose the optimizer
        optimizer = optim.SGD([
            {'params': model.parameters()},
            {'params': redshfit_regression.parameters()}
            ],lr=lr,weight_decay=0.005,momentum=0.6)  #0.05 0.5
        
        #Mix Precision To Train The Model
        scaler = torch.cuda.amp.GradScaler()
        #scheduler=lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=0.05)
        scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.05, patience=5, verbose=True)
        
        earlyStopping = EarlyStopping('./unify1_weights/{}/'.format(str(lr)),optimizer)

        #train the model
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            #with torch.autograd.detect_anomaly():
            if modal=='all':
                l1=train_one_epoch(train_loader,model,redshfit_regression,optimizer,criterion,scaler)
            else:
                l1=train_one_modal(train_loader,model,redshfit_regression,optimizer,criterion,scaler,modal)
            
            if not os.path.exists('./unify1_weights/{}'.format(str(lr))):
                os.mkdir('./unify1_weights/{}'.format(str(lr)))

            state={
            'A': model.state_dict(),
            'B': redshfit_regression.state_dict()
            }

            torch.save(state,'./unify1_weights/{}/{}_{}.pth'.format(str(lr),str(lr),str(epoch)))
            print('save the model ./unify1_weights/{}/{}_{}.pth'.format(str(lr),str(lr),str(epoch)))
            
            
            if modal=='all':
                l2=evaluate(val_loader,model,redshfit_regression,criterion)
                earlyStopping(l2[0],state,epoch)
            else:
                l2=evaluate_one_modal(val_loader,model,redshfit_regression,criterion,modal)
                earlyStopping(l2[0],state,epoch)

            if earlyStopping.early_stop:
                print("earlyStopping now!")
                break

            writer.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
            writer.add_scalar('train/mean_loss',l1[0],epoch)
            if modal=='all':
                writer.add_scalar('train/c1_loss',l1[1],epoch)
                writer.add_scalar('train/c2_loss',l1[2],epoch)
                writer.add_scalar('train/c3_loss',l1[3],epoch)
                writer.add_scalar('train/c4_loss',l1[4],epoch)
                writer.add_scalar('train/contrast_loss',l1[5],epoch)
                writer.add_scalar('train/z2_loss',l1[6],epoch)
                writer.add_scalar('train/z3_loss',l1[7],epoch)
                writer.add_scalar('train/z4_loss',l1[8],epoch)
                writer.add_scalar('train/z1_loss',l1[9],epoch)
            
            writer.add_scalar('val/mean_loss',l2[0],epoch)
            if modal=='all':
                writer.add_scalar('val/c1_loss',l2[1],epoch)
                writer.add_scalar('val/c2_loss',l2[2],epoch)
                writer.add_scalar('val/c3_loss',l2[3],epoch)
                writer.add_scalar('val/c4_loss',l2[4],epoch)
                writer.add_scalar('val/contrast_loss',l2[5],epoch)
                writer.add_scalar('val/z1_loss',l2[6],epoch)
                writer.add_scalar('val/z2_loss',l2[7],epoch)
                writer.add_scalar('val/z3_loss',l2[8],epoch)
                writer.add_scalar('val/z4_loss',l2[9],epoch)


            scheduler.step(l1[0])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,default='SDSS',help='choose the kind of data to train the model')
    parser.add_argument('--task', type=str,default='ESTIMATION',help='choose the task of our model')
    parser.add_argument('--bands', type=int,default=0,help='choose the bands of SkyMapper due to the lack of mag')
    parser.add_argument('--modal', type=str,default='photo',help='choose the modal to train our model photo,img,all etc')
    opt = parser.parse_args()
    main(opt)
