# model.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_bird_classifier_model(num_classes, pretrained=True):
    """
    Returns a pre-trained ResNet18 model modified for bird sound classification.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
        
    model = models.resnet18(weights=weights)

    # Modify the final fully connected layer for the specified number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

# Example of how to get the model:
if __name__ == '__main__':
    import config as config
    model = get_bird_classifier_model(num_classes=config.NUM_CLASSES)
    print("Model loaded successfully:")
    print(model.fc) # Print the last layer to verify
    # Dummy input to test
    dummy_input = torch.randn(1, 3, config.N_MELS, config.FIXED_LENGTH_FRAMES)
    output = model(dummy_input)
    print("Output shape:", output.shape)