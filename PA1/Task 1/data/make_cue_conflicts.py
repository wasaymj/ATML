import os
import json
import torch
import torch.nn as nn
import subprocess
import numpy as np

# AdaIN code and make_cue_conflicts logic

def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    std  = feat.view(N, C, -1).var(dim=2).add_(eps).sqrt_().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std

def adain_transform(content_feat, style_feat):
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    normalized = (content_feat - c_mean.expand_as(content_feat)) \
                  / c_std.expand_as(content_feat)
    return normalized * s_std.expand_as(content_feat) \
                      + s_mean.expand_as(content_feat)

def get_adain_models(vgg_path, decoder_path, device):
    vgg_enc = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),   nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),  nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),nn.ReLU(),
    )

    adain_dec = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64,  (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64,  64,  (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64,   3,  (3, 3)),
    )

    vgg_enc.load_state_dict(torch.load(vgg_path, map_location='cpu'), strict=False)
    adain_dec.load_state_dict(torch.load(decoder_path, map_location='cpu'))

    vgg_enc = vgg_enc.to(device).eval()
    adain_dec = adain_dec.to(device).eval()
    for p in list(vgg_enc.parameters()) + list(adain_dec.parameters()):
        p.requires_grad = False
        
    return vgg_enc, adain_dec

@torch.no_grad()
def stylize(content, style, vgg_enc, adain_dec, device, alpha=1.0):
    cf = vgg_enc(content.to(device))
    sf = vgg_enc(style.to(device))
    t  = adain_transform(cf, sf)
    t  = alpha * t + (1 - alpha) * cf
    return adain_dec(t).clamp(0, 1).cpu()

def visual_reject(img: torch.Tensor):
    if torch.isnan(img).any() or torch.isinf(img).any():
        return True, "NaN/Inf"
    std = img.std().item()
    if std < 0.05:
        return True, f"std={std:.3f}<0.05"
    mean = img.mean().item()
    if mean < 0.05 or mean > 0.95:
        return True, f"mean={mean:.3f} out of [0.05,0.95]"
    sat = ((img > 0.99) | (img < 0.01)).float().mean().item()
    if sat > 0.05:
        return True, f"sat_frac={sat:.3f}>0.05"
    return False, ""
