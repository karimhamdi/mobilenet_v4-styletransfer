import pdb
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchvision.models as mdl
import timm
from torchvision import transforms as tf
from torchvision.utils import save_image

from PIL import Image
from tqdm import trange

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



device = torch.device(args.device)


def calculate_ssim(img1, img2):
    if img1.shape != img2.shape:
        img2 = resize(img2, img1.shape)
    return ssim(img1[0], img2[0], channel_axis=0, data_range=max(img1.max()-img1.min(), img2.max()-img2.min()))

def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=1)

def calculate_content_loss(c_feats, st_feats):
    return F.mse_loss(c_feats[2], st_feats[2], reduction="mean").item()

def gram_matrix(x):
    a, b, c, d = x.shape
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def calculate_style_loss(s_feats, st_feats):
    s_loss = 0
    for ix in range(len(s_feats)):
        s_loss += F.mse_loss(
            gram_matrix(s_feats[ix]),
            gram_matrix(st_feats[ix]),
        )
    return s_loss.item()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GhostStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('ghostnet_100', pretrained=True)
        blocks = list(model.children())[:-4]
        self.net = nn.ModuleList(blocks)
    
    def forward(self, x: torch.Tensor):
        emb = []
        for block in self.net:
            x = block(x)
            emb.append(x)
        return emb


class MobStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = mdl.mobilenet_v2(True).features[:15]
        blocks = []
        blocks += [model[:2]]
        blocks += [model[2]]
        blocks += [model[3:5]]
        blocks += [model[5:8]]
        blocks += [model[8:]]
        self.net = nn.ModuleList(blocks)
    
    def forward(self, x: torch.Tensor):
        emb = []
        for block in self.net:
            x = block(x)
            emb.append(x)
        return emb


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_gm = None

    def gram_matrix(self, x):
        a, b, c, d = x.shape
        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, c_feats, s_feats, st_feats):
        # 应用WCT
        if args.wct:
            wct_transformed_feats = [wct(c, s) for c, s in zip(c_feats, s_feats)]
            c_loss = F.mse_loss(wct_transformed_feats[2][0], st_feats[2], reduction="mean")
        else:
            c_loss = F.mse_loss(c_feats[2], st_feats[2], reduction="mean")
        s_loss = 0
        for ix in range(len(s_feats)):
            s_loss += F.mse_loss(
                self.gram_matrix(s_feats[ix]),
                self.gram_matrix(st_feats[ix]),
            )
        return c_loss + (1e7 * s_loss)


def load_image(img_path: str, imsize: int = 256):
    image = Image.open(img_path)
    loader = tf.Compose([
        tf.Resize((imsize, imsize)),  # scale imported image
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = loader(image).unsqueeze(0)
    return image


def inv_normz(img):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(img.device)
    mean = (
        torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(img.device)
    )
    out = torch.clamp(img * std + mean, 0, 1)
    return out


def run_style_transfer(c_img_path: str, s_img_path: str, out_path: str = "./result.jpg",iters: int = 1000, imsize: int = 512):
    """
    run_style_transfer Neural Style Transfer using MobileNet V2.

    Args:
        c_img_path ([str]): content image path
        s_img_path ([str]): style image path
        out_path (str, optional): output image save path. Defaults to "./out.jpg".
        iters (int, optional): num. optimization iters. Defaults to 500.
        imsize (int, optional): image size. Defaults to 128.
    """
    c_img = load_image(c_img_path, imsize).to(device)
    s_img = load_image(s_img_path, imsize).to(device)
    out = c_img.clone().to(device)
    # out = torch.randn(c_img.shape, requires_grad=True)
    out.requires_grad = True
    opt = optim.LBFGS([out.requires_grad_()])
    if args.backbone == "mobilenet":
        model = MobStyleModel().eval().to(device)
    elif args.backbone == "ghostnet":
        model = GhostStyleModel().eval().to(device)
    # print(model)
    
    if args.prune_channels:
        prune_channels(model, ratio=0.2)

    loss_fn = PerceptualLoss().to(device)
    loop = trange(iters, desc="Optimizing: ")
    with torch.no_grad():
        s_feats = model(s_img)
        c_feats = model(c_img)
    for _ in loop:
        def closure():
            opt.zero_grad()
            st_feats = model(out)
            loss = loss_fn(c_feats, s_feats, st_feats)
            loop.set_postfix({"Loss: ": loss.item()})
            loss.backward()
            return loss
        opt.step(closure)
    out = inv_normz(out)
    st_feats = model(out)
    save_image(out, out_path, nrow=1)
    # Calculate metrics
    original_img = load_image(c_img_path, imsize).to(device)
    st_img = out.clone().detach()

    ssim_val = calculate_ssim(original_img.cpu().numpy(), st_img.cpu().numpy())
    psnr_val = calculate_psnr(original_img.cpu().numpy(), st_img.cpu().numpy())
    content_loss_val = calculate_content_loss(c_feats, st_feats)
    style_loss_val = calculate_style_loss(s_feats, st_feats)

    print("Done...!")
    return ssim_val, psnr_val, content_loss_val, style_loss_val


def prune_channels(model, ratio=0.2):
    """
    Prune channels based on the L1 norm of weights.
    Args:
    - model: the model to be pruned
    - ratio: the ratio of channels to be pruned
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Calculate the L1 norm for each channel
            l1_norm = m.weight.data.abs().sum(dim=(1, 2, 3))
            # Determine the threshold for pruning
            num_channels_to_prune = int(len(l1_norm) * ratio)
            if num_channels_to_prune == 0:
                continue
            _, indices = torch.topk(l1_norm, num_channels_to_prune, largest=False)
            # Set the weights of the channels to be pruned to zero
            m.weight.data[indices] = 0
            if m.bias is not None:
                m.bias.data[indices] = 0


def wct(content_feat, style_feat, alpha=0.6):
    # Whiten
    content_mean, content_std = torch.mean(content_feat, 1), torch.std(content_feat, 1)
    style_mean, style_std = torch.mean(style_feat, 1), torch.std(style_feat, 1)
    content_feat = (content_feat - content_mean.unsqueeze(1).unsqueeze(2)) / content_std.unsqueeze(1).unsqueeze(2)
    content_feat = content_feat * style_std.unsqueeze(1).unsqueeze(2) + style_mean.unsqueeze(1).unsqueeze(2)
    # Mix
    transformed_feat = alpha * content_feat + (1.0 - alpha) * content_feat
    return transformed_feat




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--backbone", type=str, default="ghostnet")
    parser.add_argument("--prune_channels", type=int, default=1)
    parser.add_argument("--wct", type=int, default=1)
    args = parser.parse_args()

    import os
    input_files = os.listdir("imgs/inputs")
    style_files = os.listdir("imgs/styles")
    N = min(len(input_files), len(style_files))

    ssim_vals, psnr_vals, content_losses, style_losses = [], [], [], []
    for i in range(1, N+1):
        input_path = f"imgs/inputs/input{i}.jpg"
        style_path = f"imgs/styles/style{i}.jpg"
        output_path = f"imgs/outputs/output{i}.jpg"
        # try:
        ssim_val, psnr_val, content_loss_val, style_loss_val = run_style_transfer(input_path, style_path, output_path)
        # except Exception as e:
            # print(e)
        ssim_vals.append(ssim_val)
        psnr_vals.append(psnr_val)
        content_losses.append(content_loss_val)
        style_losses.append(style_loss_val)

    # Calculate average metrics
    avg_ssim = sum(ssim_vals) / N
    avg_psnr = sum(psnr_vals) / N
    avg_content_loss = sum(content_losses) / N
    avg_style_loss = sum(style_losses) / N

    print(f"Average SSIM: {avg_ssim}")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average Content Loss: {avg_content_loss}")
    print(f"Average Style Loss: {avg_style_loss}")
