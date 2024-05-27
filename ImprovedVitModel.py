from torch import nn
import torch
import torch.nn as nn
from einops import rearrange
from res_model import ResNet101
#1x1 Conv
def conv_1x1_bn(inp,oup):
    return nn.Sequential(
    nn.Conv2d(inp,oup,kernel_size=1,stride=1,padding=0,bias=False),
    nn.BatchNorm2d(oup),
    nn.SiLU()
    )

#count the sum of parameters

from functools import reduce

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('The sum of the model size is：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

#IVB
class IVBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2,4]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

#Vision Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads).contiguous(), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)').contiguous()
        return self.to_out(out)

#transformer输入数据处理，对于任意输入为(B,C,H,W)进行切割为(B,(NH，NW)，(PH,PW），C)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class BranchBlock(nn.Module):
    def __init__(self,inp,oup,residual=True):
        super().__init__()
        self.residual = residual
        # 第一个分支 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(inp,inp//2, kernel_size=1),
            nn.BatchNorm2d(oup//2),
            nn.ReLU(inplace=True)
        )
        # 第二个分支 1x1卷积+3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp, inp//2, kernel_size=1),
            nn.BatchNorm2d(inp//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp//2, oup//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(oup//2),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        if self.residual == True:
           x = x + torch.cat((out1,out2), dim=1)
        return x


#Feature Fusion Module
class SENet(nn.Module):
    def __init__(self,channel,r=4):
        super().__init__()
        self.SE=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(channel,channel//r,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//r,channel,kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.SE(x)

class FeatureFusion(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.SENet=SENet(channel)

    def forward(self,x,y):
        z=x+y
        z=self.SENet(z)
        return x*z+y*(1-z)

#calculate local and global separately
class ImprovedVitBlock(nn.Module):
    def __init__(self,dim,depth,channel,patch_size,mlp_dim,dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        
        self.local=BranchBlock(channel,channel)

        self.conv1 = conv_1x1_bn(channel,dim)
        self.conv2 = conv_1x1_bn(dim,channel)

        self.globalFeature = Transformer(dim,depth,4,8,mlp_dim, dropout)

        self.aff = FeatureFusion(channel)
    def forward(self,x):
        
        # Local Features
        y = self.local(x)

        # Global Features
        _ , _, h , w=x.shape
        x = self.conv1(x)
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw).contiguous()
        x = self.globalFeature(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw).contiguous()
        x = self.conv2(x)

        # Attention Feature Fusion
        x = self.aff(x,y)

        return x

#Input Image 5x64x64
class ImprovedVit(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih , iw = image_size
        ph , pw = patch_size
        assert ih % ph == 0 and iw % pw ==0

        #layer number of VIT
        L = [2 , 3 , 3 , 4]

        self.conv1 = conv_1x1_bn(5,channels[0])

        self.ivbs = nn.ModuleList([])
        self.ivbs.append(IVBlock(channels[0],channels[1],1,expansion))
        self.ivbs.append(IVBlock(channels[1],channels[2],1,expansion))
        self.ivbs.append(IVBlock(channels[2],channels[3],1,expansion))
        self.ivbs.append(IVBlock(channels[3],channels[4],1,expansion))

        self.ivbs.append(IVBlock(channels[4],channels[5],2,expansion))
        self.ivbs.append(IVBlock(channels[5],channels[6],2,expansion))
        self.ivbs.append(IVBlock(channels[6],channels[7],2,expansion))
        self.ivbs.append(IVBlock(channels[7],channels[8],2,expansion))

        self.vit = nn.ModuleList([])
        self.vit.append(ImprovedVitBlock(dims[0],L[0],channels[5],patch_size,int(dims[0]*2)))
        self.vit.append(ImprovedVitBlock(dims[1],L[1],channels[6],patch_size,int(dims[1]*3)))
        self.vit.append(ImprovedVitBlock(dims[2],L[2],channels[7],patch_size,int(dims[2]*4)))
        self.vit.append(ImprovedVitBlock(dims[3],L[3],channels[8],patch_size,int(dims[3]*4)))


        self.conv2 = conv_1x1_bn(channels[8],channels[9])

        self.pool  = nn.AvgPool2d(4,1)

        self.fc    = nn.Linear(channels[-1], num_classes,bias=False)#channels[-1], num_classes,bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.ivbs[0](x)
        x = self.ivbs[1](x)
        x = self.ivbs[2](x)
        x = self.ivbs[3](x)
        
        x = self.ivbs[4](x)
        x = self.vit[0](x)
        
        x = self.ivbs[5](x)
        x = self.vit[1](x)

        x = self.ivbs[6](x)
        x = self.vit[2](x)

        x = self.ivbs[7](x)
        x = self.vit[3](x)

        x = self.conv2(x)

        x = self.pool(x)

        x = x.view(-1,x.shape[1])

        x = self.fc(x)

        return x

def vit_demo(img_size=(64,64),num_classes=1):
    dims = [80,96,128,196]  # 144
    channels = [8,16,32,32,48,48,64,128,196,256]  #[8,16,32,32,48,48,64,64,96,96,288]
    return ImprovedVit((img_size[0], img_size[1]), dims, channels, num_classes=num_classes)

if __name__=='__main__':
    v=ResNet101() #vit_demo()
    print(getModelSize(v))
