import torch
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from utils import split_dataset1
from astropy.io import fits
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch.nn.functional as F

class SkyMapperDataset(Dataset):
    def __init__(self,skyMapperPaths,photo,color,labels,extension,contrastClass,types='ESTIMATION',bands=0):
        self.skyMapperPaths = skyMapperPaths
        self.y = labels
        self.bands = bands
        self.extension = extension
        self.contrastClass = F.one_hot(torch.tensor(contrastClass), num_classes=11).to(torch.float32)
        self.photo = torch.tensor(photo).to(torch.float32).unsqueeze(-1)
        self.color = torch.tensor(color).to(torch.float32).unsqueeze(-1)
        
        if types=='CLASSIFICATION':
           label_encoder = LabelEncoder()
           self.y = label_encoder.fit_transform(self.y)

        elif types=='ESTIMATION':
           self.y = torch.tensor(self.y).to(torch.float32).unsqueeze(-1)

    def __preprocess_array__(self,array):
        # 获取数组的形状
        rows, cols = array.shape

        # 如果数组形状小于 (64, 64)，进行填充
        if rows < 64 or cols < 64:
            # 计算需要进行的边缘填充的行和列数
            pad_rows = max(0, 64 - rows)
            pad_cols = max(0, 64 - cols)

            # 使用 np.pad 进行边缘填充
            array = np.pad(array, ((0, pad_rows), (0, pad_cols)), mode='constant')

        # 如果数组形状大于 (64, 64)，进行裁剪
        elif rows > 64 or cols > 64:
            # 裁剪为 (64, 64)
            array = array[:64, :64]

        return array

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,item):
        labels = self.y[item]
        photo = self.photo[item,self.bands:]
        color = self.color[item,self.bands:]
        wisePath = os.sep.join(('SkyFullImage','_'.join(('SKY',str(self.extension.iloc[item,0]).replace('b','').replace('\'','').strip(),str(self.extension.iloc[item,1])))))+'.mat'
        images = joblib.load(wisePath)

        return photo,color,images[self.bands:,:,:],labels,self.contrastClass[item]

    def __getitem1__(self,item):
        skyMapperPrePath = './SKYMAPPER/'
        wisePrePath = './ZHANGWISE_IMAGE/'
        images = None
        fake_image = np.zeros((10, 64, 64))        
        # 合并skyMapper图像
        for imgPath in self.skyMapperPaths[item].split(','):
            imgPath = skyMapperPrePath+imgPath.strip()
            try:
                img = fits.open(imgPath)[0].data
            except Exception as e:
                os.remove(imgPath)
                print(imgPath)
                print(e)
                return fake_image

            img = self.__preprocess_array__(img)
            img = np.expand_dims(img, axis=0)
            
            if images is None:
               images = img
            else:
               images = np.vstack((images,img))
        
        # 合并WISE图像
        wisePath = os.sep.join((wisePrePath,'_'.join(('WISE',str(self.extension.iloc[item,0]).replace('b','').replace('\'','').strip(),str(self.extension.iloc[item,1])))))+'.mat'
        wiseImage = joblib.load(wisePath)
        images = np.vstack((images,wiseImage))
        images = torch.tensor(images).float()
        # 图像填充缺省值
        images = torch.where(torch.isnan(images),torch.full_like(images, 0),images)
        
        # 来自不同巡天的图像进行标准化
        # 获取每个矩阵的均值和标准差
        means = torch.mean(images, dim=(1, 2), keepdim=True)
        stds = torch.std(images, dim=(1, 2), keepdim=True)
        
        # 对每个矩阵进行标准化
        images = (images - means) / stds
        
        #获取标签值
        labels = self.y[item]
        
        #photo
        photo = self.photo[item]
        
        color = self.color[item]
        
        return photo,color,images,labels,self.contrastClass[item]

if __name__=='__main__':
    skys=pd.read_csv('skysna.csv')
    # 使用apply和lambda拼接多列的值
    skys['u-v']=skys['UPSF']-skys['VPSF']
    skys['v-g']=skys['VPSF']-skys['GPSF']
    skys['g-r']=skys['GPSF']-skys['RPSF']
    skys['r-i']=skys['RPSF']-skys['IPSF']
    skys['i-z']=skys['IPSF']-skys['ZPSF']
    skys['z-w1']=skys['ZPSF']-skys['W1MAG']
    skys['w1-w2']=skys['W1MAG']-skys['W2MAG']
    skys['w2-w3']=skys['W2MAG']-skys['W3MAG']
    skys['w3-w4']=skys['W3MAG']-skys['W4MAG']


    sp = SkyMapperDataset(skys['image_name_y'].to_numpy(),skys[['UPSF', 'VPSF', 'GPSF', 'RPSF','IPSF', 'ZPSF', 'W1MAG', 'W2MAG', 'W3MAG', 'W4MAG']].to_numpy(),skys[['u-v','v-g','g-r','r-i','i-z','z-w1','w1-w2','w2-w3','w3-w4']].to_numpy(),skys['CLASS'].to_numpy(),skys[['OBSID','Z']],skys['contrast_class'].to_numpy(),types='class')
    
    #train_loader = torch.utils.data.DataLoader(sp,
    #                                           batch_size=128,
    #                                           pin_memory=False,
    #                                           num_workers=6)
    sp1,sp2=split_dataset1(sp)
    train_loader = torch.utils.data.DataLoader(sp2,
                                               batch_size=2048*2,
                                               pin_memory=True,
                                               num_workers=6)
    for step, data in enumerate(tqdm(train_loader)):
        print(data[3])
        pass
