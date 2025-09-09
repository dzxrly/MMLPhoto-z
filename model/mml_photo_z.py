import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.resnet import ResNet101


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_in, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_in)
        self.layer_norm = nn.LayerNorm(dim_in, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


class FusionSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_size, num_heads=8):
        super(FusionSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, hidden_size)
        self.value = nn.Linear(input_dim, hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        # -1 means to the features number
        query = (
            self.query(inputs)
            .view(batch_size, -1, self.num_heads, self.head_size)
            .transpose(1, 2)
        )
        value = (
            self.value(inputs)
            .view(batch_size, -1, self.num_heads, self.head_size)
            .transpose(1, 2)
        )

        attention_scores = torch.matmul(query, query.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32)
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = (
            torch.matmul(attention_weights, value)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.hidden_size)
        )
        output = self.layer_norm(context)
        return output


class PhotoEncoder(nn.Module):
    """Network to encoder photodata feature representation"""

    def __init__(self, input_dim=14, output_dim=256):
        super(PhotoEncoder, self).__init__()
        self.fs1 = FusionSelfAttention(input_dim=1, hidden_size=32)
        self.fs2 = FusionSelfAttention(input_dim=32, hidden_size=8)
        self.feedforward = FeedForward(32, 64)
        self.fc1 = nn.Linear(input_dim // 2 + (input_dim // 2 + 1) * 8, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.bn1 = nn.BatchNorm1d(input_dim // 2 + 1)  # ,momentum=0.5)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)  # ,momentum=0.5)

    def forward(self, x1, x2):
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x1 = self.fs1(x1)
        x1 = self.feedforward(x1)
        x1 = self.fs2(x1)
        x1 = torch.flatten(x1, start_dim=1)
        x2 = x2.squeeze(-1)
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class PhotoDecoder(nn.Module):
    """Network to decode photodata representation"""

    def __init__(self, input_dim=256, output_dim=256, hidden_dim=512):
        super(PhotoDecoder, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)  # ,momentum=0.5)
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x), 0.2)
        out = F.relu(self.denseL2(out), 0.2)
        return out


class ImgEncoder(nn.Module):
    def __init__(self, input_dim=5, output_dim=256):
        super(ImgEncoder, self).__init__()
        self.vit = ResNet101(
            input_dim
        )  # vit_demo(num_classes=output_dim)  #ResNet101()  #vit_demo(num_classes=output_dim)

    def forward(self, x):
        y = self.vit(x)
        return y


class ImgDecoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, hidden_dim=512):
        super(ImgDecoder, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)  # ,momentum=0.5)
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.denseL3 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        out = F.relu(self.denseL2(out))
        out = F.relu(self.denseL3(out))
        return out


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class ModalClassifier(nn.Module):
    """Network to discriminate modalities"""

    def __init__(self, input_dim=40):
        super(ModalClassifier, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)  # ,momentum=0.5)
        self.bn2 = nn.BatchNorm1d(input_dim // 4)
        self.bn3 = nn.BatchNorm1d(input_dim // 16)

        self.denseL1 = nn.Linear(input_dim, input_dim // 4)
        self.denseL2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.denseL3 = nn.Linear(input_dim // 16, 2)

    def forward(self, x):
        """Gradient Reverse"""
        x = ReverseLayerF.apply(x, 1.0)
        x = self.bn1(x)
        out = F.relu(self.denseL1(x))
        out = self.bn2(out)
        out = F.relu(self.denseL2(out))
        out = self.bn3(out)
        out = F.relu(self.denseL3(out))
        return out


class RegressionClassifier(nn.Module):
    """Network to estimate the redshift"""

    def __init__(self, input_dim=512, num_classes=4):
        super(RegressionClassifier, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)  # ,momentum=0.5)
        self.bn2 = nn.BatchNorm1d(input_dim // 4)
        self.bn3 = nn.BatchNorm1d(input_dim // 16)
        self.bn4 = nn.BatchNorm1d(input_dim // 32)

        self.denseL1 = nn.Linear(input_dim, input_dim // 4)
        self.denseL2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.denseL3 = nn.Linear(input_dim // 16, input_dim // 32)
        self.denseL4 = nn.Linear(input_dim // 32, num_classes)

    def forward(self, x):
        x = self.bn1(x)
        out = F.relu(self.denseL1(x))
        out = self.bn2(out)
        out = F.relu(self.denseL2(out))
        out = self.bn3(out)
        out = F.relu(self.denseL3(out))
        out = self.bn4(out)
        out = F.relu(self.denseL4(out))
        return out


class ContrastNN(nn.Module):
    def __init__(self, img_input_dim=5, photo_input_dim=9, num_classes=1):
        super(ContrastNN, self).__init__()
        self.img_encoder = ImgEncoder(input_dim=img_input_dim)
        self.photo2img = PhotoDecoder()
        self.photo_encoder = PhotoEncoder(input_dim=photo_input_dim)
        self.img2photo = ImgDecoder()

        self.img_judge = ModalClassifier(256)
        self.photo_judge = ModalClassifier(256)

        self.num_classes = num_classes

    def forward(self, x1, x2, image):
        photo_feature = self.photo_encoder(x1, x2)
        img_feature = self.img_encoder(image)
        img2photo_feature = self.img2photo(img_feature)
        photo2img_feature = self.photo2img(photo_feature)
        img2img_judge = self.img_judge(img_feature)
        img2photo_judge = self.photo_judge(img2photo_feature)
        photo2photo_judge = self.photo_judge(photo_feature)
        photo2img_judge = self.img_judge(photo2img_feature)

        # print(
        #     "photo_feature: {}\n".format(photo_feature.shape),
        #     "img_feature: {}\n".format(img_feature.shape),
        #     "img2photo_feature: {}\n".format(img2photo_feature.shape),
        #     "photo2img_feature: {}\n".format(photo2img_feature.shape),
        #     "img2img_judge: {}\n".format(img2img_judge.shape),
        #     "img2photo_judge: {}\n".format(img2photo_judge.shape),
        #     "photo2img_judge: {}\n".format(photo2img_judge.shape),
        #     "photo2photo_judge: {}\n".format(photo2photo_judge.shape),
        # )

        return (
            img_feature,
            photo_feature,
            img2photo_feature,
            photo2img_feature,
            img2img_judge,
            photo2img_judge,
            img2photo_judge,
            photo2photo_judge,
        )


class BuildModel(nn.Module):

    def __init__(
        self,
        img_input_channel: int,
        mags_input_dim: int,
        extractor_out_dim: int,
        out_gaussian_groups: int,
    ):
        super().__init__()
        self.out_dim = out_gaussian_groups * 3
        self.extractor = ContrastNN(
            img_input_dim=img_input_channel,
            photo_input_dim=mags_input_dim,
        )
        self.z1_regressor = RegressionClassifier(
            input_dim=extractor_out_dim, num_classes=self.out_dim
        )
        self.z2_regressor = RegressionClassifier(
            input_dim=extractor_out_dim, num_classes=self.out_dim
        )
        self.z3_regressor = RegressionClassifier(
            input_dim=extractor_out_dim, num_classes=self.out_dim
        )
        self.z4_regressor = RegressionClassifier(
            input_dim=extractor_out_dim, num_classes=self.out_dim
        )

    def forward(
        self, mags: torch.Tensor, mags_diff: torch.Tensor, img: torch.Tensor
    ) -> tuple:
        (
            img_feature,
            mags_feature,
            img2mags_feature,
            mags2img_feature,
            img2img_judge,
            mags2img_judge,
            img2mags_judge,
            mags2mags_judge,
        ) = self.extractor(mags, mags_diff, img)
        z1 = self.z1_regressor(torch.cat((mags_feature, img_feature), dim=1))
        z2 = self.z2_regressor(torch.cat((mags_feature, mags2img_feature), dim=1))
        z3 = self.z3_regressor(torch.cat((img2mags_feature, img_feature), dim=1))
        z4 = self.z4_regressor(torch.cat((img2mags_feature, mags2img_feature), dim=1))
        return (
            z1,
            z2,
            z3,
            z4,
            img_feature,
            mags_feature,
            img2mags_feature,
            mags2img_feature,
            img2img_judge,
            mags2img_judge,
            img2mags_judge,
            mags2mags_judge,
        )
