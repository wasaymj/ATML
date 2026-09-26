import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
import open_clip

class BackboneWrapper(nn.Module):
    def __init__(self, backbone, norm_layer, feat_dim: int, name: str):
        super().__init__()
        self.backbone  = backbone
        self.norm      = norm_layer
        self.feat_dim  = feat_dim
        self.name      = name
 
    @torch.no_grad()
    def get_features(self, x: torch.Tensor, device) -> torch.Tensor:
        x = self.norm(x.to(device))
        return self.backbone(x)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_features(x, x.device)
 
class CLIPWrapper(BackboneWrapper):
    @torch.no_grad()
    def get_features(self, x: torch.Tensor, device) -> torch.Tensor:
        x = self.norm(x.to(device))
        feat = self.backbone(x)
        return feat / feat.norm(dim=-1, keepdim=True)

class LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

def get_models(device):
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()
    for p in resnet.parameters():
        p.requires_grad = False
        
    vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity()
    vit = vit.to(device).eval()
    for p in vit.parameters():
        p.requires_grad = False
        
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
        
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]
    _CLIP_MEAN     = [0.48145466, 0.4578275,  0.40821073]
    _CLIP_STD      = [0.26862954, 0.26130258, 0.27577711]
     
    NORM = {
        "resnet": transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        "vit"   : transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        "clip"  : transforms.Normalize(mean=_CLIP_MEAN,     std=_CLIP_STD),
    }
    
    FEAT_DIM = {"resnet": 2048, "vit": 768, "clip": 512}
    
    resnet_wrapper = BackboneWrapper(resnet, NORM["resnet"], FEAT_DIM["resnet"], "resnet")
    vit_wrapper    = BackboneWrapper(vit, NORM["vit"], FEAT_DIM["vit"], "vit")
    clip_wrapper   = CLIPWrapper(clip_model.visual, NORM["clip"], FEAT_DIM["clip"], "clip")
    
    MODELS = {
        "resnet": resnet_wrapper,
        "vit"   : vit_wrapper,
        "clip"  : clip_wrapper,
    }
    
    return MODELS, clip_model, FEAT_DIM
