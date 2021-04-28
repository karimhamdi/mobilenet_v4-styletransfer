import pdb
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchvision.models as mdl
from torchvision import transforms as tf
from torchvision.utils import save_image

from PIL import Image
from tqdm import trange
import typer

class MobStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = mdl.mobilenet_v2(True).features[:15]
        # * downsamples: 0, 2, 4, 7, 14
        # * content: 4
        # * style: 1, 2, 4, 7, 14
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

def run_style_transfer(c_img_path: str, s_img_path: str, out_path: str = "./out.jpg",iters: int = 500, imsize: int = 128):
    """
    run_style_transfer Neural Style Transfer using MobileNet V2.

    Args:
        c_img_path ([str]): content image path
        s_img_path ([str]): style image path
        out_path (str, optional): output image save path. Defaults to "./out.jpg".
        iters (int, optional): num. optimization iters. Defaults to 500.
        imsize (int, optional): image size. Defaults to 128.
    """
    
    c_img = load_image(c_img_path, imsize)
    s_img = load_image(s_img_path, imsize)
    # out = torch.randn(c_img.shape, requires_grad=True)
    out = c_img.clone()
    out.requires_grad = True
    
    opt = optim.LBFGS([out.requires_grad_()])
    
    model = MobStyleModel().eval()
    loss_fn = PerceptualLoss()

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
    save_image(out, out_path, nrow=1)
    print("Done...!")


if __name__ == "__main__":
    typer.run(run_style_transfer)
