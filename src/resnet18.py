# src/simple_model.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_bird_model(num_classes, pretrained=True):
    """
    Returns a ResNet18 model adapted for bird sound classification.
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use ImageNet pre-trained weights.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        print("Loading ResNet18 with pre-trained ImageNet weights.")
    else:
        weights = None
        print("Loading ResNet18 with random weights.")
        
    model = models.resnet18(weights=weights)

    # The input to ResNet18's first conv layer is:
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Our spectrograms will be prepared as (3, N_MELS, SPEC_FRAMES), so this is compatible.

    # Modify the final fully connected layer (classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"ResNet18 final layer replaced: {num_ftrs} in-features, {num_classes} out-features.")

    return model
