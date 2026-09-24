import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet_cifar(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Returns a CIFAR-appropriate ResNet-18.
    - Replaces conv1 with 3x3, stride=1, padding=1
    - Removes maxpool
    - Replaces fc layer with num_classes
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Modify conv1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove maxpool by replacing it with Identity
    model.maxpool = nn.Identity()
    
    # Replace FC
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

class SplitResNet(nn.Module):
    """
    Wrapper around ResNet18 that exposes features before the FC layer.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = get_resnet_cifar(num_classes=num_classes, pretrained=False)
        self.in_features = self.model.fc.in_features
        
    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        f = torch.flatten(x, 1)
        
        logits = self.model.fc(f)
        
        if return_features:
            return logits, f
        return logits

    def forward_layer2(self, x):
        """Returns features after layer2 (for Manifold Mixup)."""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        return x
        
    def forward_from_layer3(self, x):
        """Continues forward pass from layer3 (for Manifold Mixup)."""
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        f = torch.flatten(x, 1)
        
        logits = self.model.fc(f)
        return logits, f
